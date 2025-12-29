# The emergence of Large Language Models (LLM) as a tool in literature reviews: an  LLM automated systematic review 
Dmitry Scherbakov, PhD1,*, Nina Hubig, PhD 1,2,*, Vinita Jansari, PhD2, Alexander 
Bakumenko, MSc2 , Leslie A. Lenert, MD1
Bakumenko, MSc2 , Leslie A. Lenert, MD1  
 Biomedical Informatics Center, Department of Public Health Sciences, Medical University of  South Carolina (MUSC), Charleston, South Carolina, USA.  2 Clemson University, School of Computing , Charleston, South Carolina, USA.  * Authors with equal contribution. 
Corresponding author: Leslie Lenert (lenert@musc.edu)  Address: 22 WestEdge Street, Suite 200, Room WG213, Charleston, South Carolina, 29403, USA 
Manuscript word count: 2987 (excluding title page, abstract, references, tables,  acknowledgements, funding, data availability, competing interests, and author contributions  statements)  Abstract word count: 301 (excluding keywords)  
Highest academic degrees of authors (in order):  DS : PhD, Postdoctoral scholar  NH : PhD, Assistant professor  VJ : PhD, Postdoctoral scholar  AB : MSc, Doctoral student  LL : MD, MS, FACP, FACMI, Professor 
This study aims to summarize the usage of Large Language Models (LLMs) in the process of  creating a scientific review. The idea behind this publication is to turn the tool “on itself”,  conducting a systematic review of research projects using LLMs for systematic and other types of reviews by using a set of LLM tools. 
Objective: This study aims to summarize the usage of Large Language Models (LLMs)  in the process of creating a scientific review. We look at the range of stages in a review that can  be automated and assess the current state-of-the-art research projects in the field.  Materials and Methods: The search was conducted in June 2024 in PubMed, Scopus,  Dimensions, and Google Scholar databases by human reviewers. Screening and extraction  process took place in Covidence with the help of LLM add-on which uses OpenAI gpt-4o model. ChatGPT was used to clean extracted data and generate code for figures in this manuscript,  ChatGPT and Scite.ai were used in drafting all components of the manuscript, except the  methods and discussion sections.  Results:  3,788 articles were retrieved, and 172 studies were deemed eligible for the final  review. ChatGPT and GPT-based LLM emerged as the most dominant architecture for review  automation (n=126, 73.2%). A significant number of review automation projects were found, but only a limited number of papers (n=26, 15.1%) were actual reviews that used LLM during their  creation. Most citations focused on automation of a particular stage of review, such as Searching  for publications (n=60, 34.9%), and Data extraction (n=54, 31.4%). When comparing pooled  performance of GPT-based and BERT-based models, the former were better in data extraction  with mean precision 83.0% (SD=10.4), and recall 86.0% (SD=9.8), while being slightly less  accurate in title and abstract screening stage (Maccuracy=77.3%, SD=13.0 vs Maccuracy=80.9%  SD=11.8).   Discussion/Conclusion: Our LLM-assisted systematic review revealed a significant  number of research projects related to review automation using LLMs.  The results looked  promising, and we anticipate that LLMs will change in the near future the way the scientific  reviews are conducted, significantly reducing the time required to generate systematic reviews of the literature and expanding how systematic reviews are used to guide science.  Keywords: Large Language Models, Review Automation, Systematic Review, Scoping  Review, Covidence. 
The abundance of scientific information available can be overwhelming, posing a  challenge for researchers to navigate relevant data. Consequently, scoping and systematic  reviews that are helping scientists synthesize the evidence have seen a significant increase over  the years. Toh & Lee noted an exponential rise in the number of scoping reviews, with 2,665  scoping reviews being published in 2020 alone, compared to less than 10 reviews annually  before 2009 1. The same trend is observed in systematic reviews and meta-analyses, for example in cardiology over 2,400 meta-analyses were published in 2019, quadruple the number from  2012 2.    A completion of a review requires substantial resources; further, there is often  unpredictable uncertainty in the amount of resources required 3. The time to complete a single  systematic review varies, but authors typically give estimates in months and even years 4.  Screening automation platforms, such as Covidence 5, facilitate systematic and scoping reviews by streamlining established guidelines, such as the Preferred Reporting Items for Systematic  Reviews and Meta-Analyses (PRISMA) and PICO (Population, Intervention, Comparison, and  Outcome) to ensure transparency and rigor in the review process 6. The  use of such platforms  may reduce the time to complete reviews by providing tools that automate key tasks, such as  removing duplicate references, generating flow-charts of the screening process, visual extraction designers, and workflows for several independent reviewers.    Although, for example, Covidence, includes features to reduce the time to complete  screening, such as key term highlighting and embedded natural processing (NLP) algorithm 7  it primarily organizes the significant manual work that is still needed from human reviewers like  screening and extraction.. Each of these steps normally requires two independent analysts, with  third optional human expert supervising the process and resolving the disagreements.    Even with two reviewers double-checking each other, as much as 3% of relevant citation are missed, and if only a single reviewer is used (for example, in rapid reviews), as many as 13% of relevant publications can be missed 8. The relatively weak performance of humans in  screening relevant articles has led some investigators to develop natural language processing  tools 9-12 to automate screening. A recent statement by the National Institute for Health and Care Excellence (NICE) highlights a big potential of AI in the systematic review process automation 
Large Language Models (LLM) recently emerged as one of the most powerful NLP tool across different ranges of tasks. By conducting this review, we wanted to evaluate the natural  extension of the use of LLMs to guide and direct the review process. Thus, this systematic  review aims to (1) summarize the current state-of-the-art research projects using LLMs to  automate the review process, (2) look at the range of review types and review stages that are  being automated, (3) assess the quality of each research project, (4) assess the performance of  LLMs used for automation.  As LLMs are used as a possible substitute for a human reviewer, the idea behind this  publication is to turn the tool “on itself”, conducting a systematic review of research projects  using LLMs for systematic reviews.  
# Methods 
The study's research plan was formulated by the author team and adjusted based on the guidance provided by the preferred reporting items for systematic review and meta-analysis protocols (PRISMA-P) 2015: elaboration and explanation 14 and the latest JBI checklist 15 for conducting systematic reviews. The review protocol was registered in the Open Science Framework (OSF) database 16.  We decided that to be included in the review, citations had to be centered around the usage of LLMs in automation of different phases of systematic review. Only English-language journal publications were considered, including, conference abstracts, and review publications that used LLMs in their creation.   Publications were excluded if they:  ●  Did not use some kind of LLM (e.g. ChatGPT, Mistral, GPT-3.5, BERT)  ●  Did not describe automation of any stage of the review process  ●  The paper was a review article itself that did not use LLM to conduct the review  ●  Full text of the article could not be retrieved or was not in English 
The initial search was conducted by a human reviewer (DS) in June 2024 in PubMed, Scopus, Dimensions, and Google Scholar databases. Table 1 presents the search strategy for the databases.   Table 1. Search strategy.  
(("large language models" OR "large language model" OR "LLM" OR "LLMs" OR "ChatGPT" OR "GPT-3" OR  "GPT-4" OR "LLaMA" OR "Mistral" OR "Mixtral" OR "BARD" OR "BERT" OR "Claude" OR "PaLM" OR  "Gemini" OR "Copilot") AND ("systematic review*" OR "scoping review*" OR "literature review*" OR "narrative  review*" OR  "umbrella review*" OR "rapid review*" OR "integrative review*" OR "evidence synthesis" OR  "meta-analysis"))   Source: Authors’ own work 
All citations were then uploaded to Covidence. The screening and extraction process took place in Covidence with the help of LLM plugin for Covidence that our team developed. This plugin is used during screening and extraction phases. The process of using LLM for screening and extraction is shown on Figure 1. 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fdbc/fdbc0b70-3076-4f44-bd14-8535c4680270.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1. LLM workflow added into Covidence for screening and extraction. Source: Authors’ own work. </div>
The developed add-on works by interacting with the Covidence platform programmatically via an intermediary software solution that was created in Python and R. The solution passes contents between Covidence and the LLM OpenAI gpt-4o model . Once the LLM generates the
response, a script automates actions in Covidence, such as clicking Include/Exclude buttons or  leaving notes.   The review process involved three stages that were automated by Covidence add-on:  abstract screening, full-text screening, and extraction. In each stage, two human reviewers  calibrated by screening a sample to refine inclusion criteria and extraction categories. They then  created and tested prompts for the LLM. LLM inference was programmed to run inference 3 times  to determine the final decision (e.g., "include" or "exclude") based on the majority vote. Three  prompts per phase are detailed in Supplementary Appendix S1.  For abstract screening, LLM and human reviewers voted for consensus, and a human expert  consensus was established. In full-text screening and extraction, a single human reviewer verified  LLM results. Extraction precision was measured, and for categories with low precision (<80%), a  manual reviewer validated LLM outputs. Benchmarks are provided in Supplementary Appendix  S2.  The data charting form for extraction were designed by human experts (DS, VJ, AB, LL,  and NH) and adopted into the LLM prompt to collect the following primary information:   ●  Author, year, title;   ●  Country and/or US state;  ●  What types of reviews were automated;  ●  Stage of review automated in the research project;   ●  LLM type used;   ●  Performance metrics reported by authors during each stage of the review. In  particular, Accuracy, Precision, Recall, Specificity, and F1 were extracted, if other  metrics were used instead, they were grouped under “Other metrics” category;  ●  Brief information on how were these performance metrics calculated;   ●  Brief information on reported timesaving;  ●  What was general opinion of the study team on the usage of LLMs in review  automation (positive, negative, or mixed) with a citation to support this viewpoint.     Human reviewers (DS, VJ, AB) performed quality assessment of given studies using a set 
Human reviewers (DS, VJ, AB) performed quality assessment of given studies using a set of selected categories from the reviewed studies and a points-based scale:  
●  Ratings of the universities where authors are affiliated (the data was link ranking 2024 17), maximum value across all co-author affiliations was use ■  Ranked 1 to 100: 2 points   ■  100-1000: 1 point  ■  >1000: 0 point  ●  Number of samples (full-texts or abstracts) that authors used to comp performance metrics:  ■  More than 200: 2 points   ■  50-200: 1 point  ■  <50: 0 points    ●  Sources of the funding of the research project (public, private or mixed)   ■  Public funding: 2 points   ■  No funding: 1 point  ■  Private funding: 0 points   ●  Impact factor of the journal  18  ■  More than 5: 2 points   ■  1 to 5: 1 point   ■  Less than 1: 0 points  ●  Is the paper an actual review which used LLM?  ■  A review: 2 points  ■  Not a review (methods paper): 1 point  ●  Were performance metrics (benchmarks) reported?   ■  2 points for reporting performance metrics   ■  0 points for no metrics 
If value in any above category could not be determined (e.g. no match for university or  impact factor, or unknown value in category), then the NA value was assigned. Based on the mean  of points across all the quality categories, studies were classified as low (<1 points), medium (1 to  1.5 points) or high quality (>=1.5 points).  An LLM tool by Google (NotebookLM, version from August 2024) along with a manual  review (DS, VJ, AB) was used to cross check the extraction results for the fields where precision 
of extraction was low (<0.8) during the benchmark. Again, ChatGPT (4o model) was used to clean the extraction data: format the case, remove duplicates, rename similar entries to a common name. The data was then manually fed into the chat window by a human reviewer (DS). Scite.ai (version from August 2024) was used to draft parts of the introduction and discussion sections, while ChatGPT was used to draft the abstract and results section of this review by generating R code snippets to produce all figures (except Figure 1 which was generated by Covidence). ChatGPT was also used to draft the text of the results section, which was then corrected by our team where needed. Human experts edited and verified the final LLM-generated draft of the manuscript.   Additionally, we report the time saving and the computational costs in Supplementary Appendix S3. We used our own time measurements and reference data from experienced reviewers to calculate time-saving 19. 
# Results
Figure 2 outlines the PRISMA article selection process for this study. Initially, 3,788 studies were identified across several databases: PubMed (n = 2,174), Scopus (n = 1,207), Dimensions (n = 356), and Google Scholar (n = 48), along with 3 additional studies from citation searching. Following the removal of 447 duplicates (1 manually and 446 by Covidence), 3,341 studies remained for the screening phase.  During the title and abstract screening process, 3,041 studies were excluded, leaving 300 studies for retrieval and full-text eligibility assessment. Out of these 300 studies, 128 were excluded for various reasons, with the most common being “The paper does not describe the automation of any stage of the review process” (n = 88). A total of 172 studies were included in the final review.   
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c7bd/c7bda891-a270-4b5b-aa17-eecdb6b1c123.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2. Flow diagram of the systematic review process. Source: Authors’ own work / Covidence.</div>
Figure 2. Flow diagram of the systematic review process. Source: Authors’ own work / Covidence. 
   Figure 3 shows the geographic distribution of studies across 43 countries. Most citations are from  the US (n=60, 34.9%), followed by Australia (n=14, 8.14%), the UK and China (n=13, 7.6%), and Germany  (n=11, 6.4%). Other notable contributors include Canada (n=7, 4.1%) and India (n=6, 3.5%). Austria,  Ireland, Italy, the Netherlands, and South Korea each contributed 4 studies (2.3%), while countries like  New Zealand, France, Japan, and others provided 3 (1.7%). The rest contributed 1–2 studies.  In the US, 47 studies had state-level data. Tennessee, New York, and Massachusetts led  with 5 citations each (10.6%), followed by California (n=4, 8.5%). North Carolina and Ohio  contributed 3 studies (6.4%), while several other states provided 2 (4.3%) or 1 (2.1%) citation 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7057/70576d66-03f3-4054-a7eb-220f15959044.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8337/833738b9-c131-4c94-8ae7-5fb9631d7331.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 3. A: Publications by country of origin; B: Publications by state in the US. Source ChatGPT-generated code using extracted data from this manuscript (Supplemental Table S7) </div>
Figure 4A shows the types of reviews discussed in automation papers. The most frequently  mentioned type is ‘Systematic Review’ (n=118, 68.6%), followed by ‘Literature/Narrative  Review’ (n=37, 21.5%) and ‘Meta-Analysis’ (n=19, 11.0%). The remaining categories include  ‘Scoping Review’ (n=8, 4.7%), ‘Other/Non-specific’ (n=14, 8.1%), and ‘Rapid Review’ (n=6,  3.5%). ‘Umbrella Review’ has a smaller representation with 2 mentions (1.2%).  Figure 4B illustrates the stages of review discussed in automation papers. The most  frequently mentioned stage is ‘Searching for publications’ (n=60, 34.9%), followed by ‘Data  extraction’ (n=54, 31.4%) and ‘Evidence synthesis/summarization’ (n=32, 18.6%). Other  categories with notable mentions include ‘Title and abstract screening’ (n=43, 25.0%), ‘Drafting  a publication’ (n=22, 12.8%), ‘Full-text screening’ (n=14, 8.1%), ‘Quality and bias assessment’  (n=12, 7.0%), ‘Publication classification’ (n=10, 5.8%), ‘Other stages’ (n=6, 3.5%), and ‘Code  and plots generation’ (n=4, 2.3%). 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/8375/8375f51a-3e56-492c-97a2-b1566b003848.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/5094/50942df1-fba3-43bf-a897-fbfe0c621b39.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 4. A. Types of review automated B. Which stages of review are automated in the pape Source: ChatGPT-generated code using extracted data from this manuscript (Supplemental Table S7) </div>
 The most frequently mentioned AI model is GPT/ChatGPT, with 126 occurrences (73.3%),  showing its widespread use (Supplemental Figure S5). BERT-based models are also notable with 32  mentions (18.6%). LLaMA/Alpaca models have 8 mentions (4.7%), followed by Google Bard/Gemini with  5 (2.9%), and Claude models with 7 (4.1%). Other models like BART (n=3, 1.7%) and Mistral (n=4, 2.3%)  are less frequent. Several models, including Bing and XLNet, have 2 mentions each (1.2%), while many  others are mentioned just once (0.6%).  Of the 172 citations, 79 (45.9%) reported common metrics like Accuracy, Precision/Recall,  and F1, while 36 (20.9%) used less common metrics like G-score and Jaccard similarity. The  remaining 57 publications (33.1%) relied on qualitative assessments.  Figure 5 shows performance metrics for GPT- and BERT-based models. GPT models had  lower accuracy in title/abstract screening (M=77.34, SD=13.06) compared to BERT models  (M=80.87, SD=11.81). However, GPT models performed better in data extraction, with precision  (M=83.07, SD=10.43) and recall (M=85.99, SD=9.82), while BERT models had lower precision  (M=61.06, SD=31.26) and similar recall (M=80.03, SD=10.09). In title/abstract screening, BERT 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/f6ef/f6ef19c9-e97b-47ce-803f-08b390180299.png" style="width: 50%;"></div>
models had higher precision (M=65.6, SD=17.65) but lower recall (M=72.93, SD=23.95) than GPT models (precision M=63.2, SD=24.34; recall M=80.42, SD=23.31). 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/612e/612eeaea-b41b-4a25-8dd7-24400df17efa.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 5. Performance metrics reported for the three most common automated stages A: for  GPT-based models. B: for BERT-based models. Source: ChatGPT-generated code using extracted data from this manuscript (Supplemental Table S7). </div>
Figure 5. Performance metrics reported for the three most common automated stages A: for  GPT-based models. B: for BERT-based models. Source: ChatGPT-generated code using extracted data from this manuscript (Supplemental Table S7). 
Majority of reviewed publications were papers describing how LLM could be used to  automate a certain phase of the review (n=146, 84.9%) (Supplemental Figure S6A). Only 26  (15.1%) papers were actual reviews conducted with some help from LLM tools. Majority of  authors were positive about the usage of LLMs in reviews (n=120, 69.8%), with 43 citations  (25.0%) containing mixed or cautious views on LLM usage (Supplemental Figure S6B). Only 9  (5.2%) study teams had negative experiences with LLM usage. Most studies had public funding  reported (n=97, 56.4%) (Supplemental Figure S6C). When considering all the factors together,  such as funding, journal impact factor, sample size, reported metrics, and others (see Methods), 72  citations (41.9%) appear to be of high quality, with 73 citations being medium quality (42.4%)  (Supplemental Figure S6D).   
Supplemental Table S7 presents the extraction table with all extracted categories across  172 citations. 
Our LLM-assisted systematic review revealed a significant number of research projects related to review automation with LLM. Indeed, other researchers have noted promising results
or LLMs in different areas, such as understanding human language and generating contextuall ppropriate responses 20-22.  
# for LLMs in different areas, such as understanding human language and generating contextual appropriate responses 20-22.  
Despite finding a significant number of projects using LLMs to automate some stages of  the review process only few papers focused on the full cycle of review automation 23,24. There  might be perceived publication barriers, for example, journals recently started to ask about LLMgenerated content, although we don’t have information on whether this leads to changes in  reviewing process. Growing number of LLM-generated papers will probably eventually change  how review is conducted (reviewers might be assisted by LLMs or review paper format could be  eventually replaced by online real-time information retrieval).  The strength of present review is in large-scale (over 3000 abstracts screened, and 172 fulltext publications eligible for extraction) automation of different stages of review, including  drafting the manuscript sections, and plot generation. Only few citations focused on automation of  full cycle of review, while most focused only on specific areas like extraction or screening,  including our own previous systematic review where GPT-3.5 was used with LDA-based topic  modelling for validation of human findings 25. In contrast, the LLM-based method that we applied  in this work demonstrated its direct applicability, by facilitating the automation of the abstract and  full-text screening, data extraction, as well as the knowledge synthesis stages, with the discussed  constraints. Furthermore, our method is domain-agnostic, thus it can be integrated into large-scale  review projects across different domains. The implications of such automation include reducing  human workload and improving overall efficiency of systematic reviews. Furthermore, such tool  in its more mature form will require less expertise from human reviewers, which could contribute  to the democratization of systematic and scoping review process, with the potential to add features  related to meta-analysis into the process.  GPT-based LLM were the most dominant type of LLM and the one that seems to show  remarkable results on the data extraction, arguably the most complex and time-consuming stage  of any review. It’s usage for literature reviews is obvious, at this moment there are little restrictions  on the type of information users can load into ChatGPT, and published papers are unlikely to  contain any sensitive information, making ChatGPT with its high-performing model and  developed API an obvious choice. At the same time smaller models like BERT, Llama or Mistral 
can be run and fine-tuned locally with much less cost, so we expect to see more automation projects with this LLM in the future.26 
# Limitations 
We used calibrated LLMs as reviewers in this project. Some extraction categories, such as performance metrics, had relatively lower accuracy, so the results of this extraction category should be taken with caution. Nevertheless, in this review LLMs achieved remarkable results in accuracy, making it possible to delegate time-consuming phases of review to LLMs. Studies generally recommend a single reviewer approach in some cases like rapid reviews27. However we believe that the LLM approach could substitute human reviewers, and human effort should be redirected to supervision of the review process.   A further limitation of this work is the simplified scoring system we introduced for research evaluation, which, using arbitrary weightings, may overlook key aspects like the novelty, robustness, and relevance of the studies. Future research should focus on improving LLM performance metrics, particularly precision and recall in lower-accuracy extraction categories. Additionally, integrating and evaluating different LLMs, possibly in combination with other AI models, should be explored to enhance performance. The short- and long-term impact of these integrations on review quality, along with ethical considerations, must also be assessed to maintain research credibility and trust. 
# Conclusion
The use of LLMs in review automation is rapidly growing, with expected radical changes in scientific evidence synthesis. LLMs are likely to significantly reduce the time needed for reviews while producing similar or higher-quality data in greater quantities than manual reviews. Research shows it is becoming increasingly difficult to distinguish between LLM-generated and human-written text.28 and the presence of LLM generated texts in scientific publications in
growing exponentially 29. To promote transparency and proper acknowledgment, researchers are  encouraged to openly disclose their use of LLMs in academic papers, providing information on  the prompts employed and the sections of text affected 30.  Despite early successes, few systematic reviews using LLMs were identified in our review.  Although still in its early stages, AI-assisted reviews are already yielding impressive results, with  growing interest as researchers develop semi-automated pipelines. However, generating  trustworthy and useful AI-driven reviews still presents both technological and ethical challenges,  particular for quantitative meta-analyses comparing treatment effects. However, the conduct of  more simple systematic reviews, such as scoping reviews, appears to be well within the capabilities  of current or near future AI methods.    
growing exponentially 29. To promote transparency and proper acknowledgment, researchers are encouraged to openly disclose their use of LLMs in academic papers, providing information on the prompts employed and the sections of text affected 30. 
Despite early successes, few systematic reviews using LLMs were identified in our review.  Although still in its early stages, AI-assisted reviews are already yielding impressive results, with  growing interest as researchers develop semi-automated pipelines. However, generating  trustworthy and useful AI-driven reviews still presents both technological and ethical challenges,  particular for quantitative meta-analyses comparing treatment effects. However, the conduct of  more simple systematic reviews, such as scoping reviews, appears to be well within the capabilities  of current or near future AI methods.    
# Author contributions
LL, NH, and DS conceived and designed the review. DS developed the LLM screening  automation add-on for Covidence. DS and VJ contributed to search strategy development. DS,  AB and VJ performed the benchmarks for LLM and designed LLM prompts. DS, AB and VJ  verified data extraction results. VJ and AB researched third-party components that were used to  create the review. AB developed the script for journal impact factor assessment. DS analyzed th data and drafted the manuscript with the help of scite.ai and ChatGPT. NH and LL found the  resources to conduct the review. All authors critically reviewed and revised the manuscript and  approved the final version for submission.  
# Funding 
This publication was supported, in part, by the National Center for Advancing  Translational Sciences of the National Institutes of Health under Grant Number UL1 TR001450. Dr. Scherbakov was supported by grant T15 LM013977, Biomedical Informatics and Data  Science for Health Equity Research (SC BIDS4Health). This publication was supported in part  by a Smart-state Chair endowment. The content is solely the responsibility of the authors and  does not necessarily represent the official views of the National Institutes of Health.  
# Conflicts of interest statement
The authors have no competing interests to declare.
1.  Toh TS, Lee JH. Statistical note: Using scoping and systematic reviews. Pediatric Critical Care  Medicine 2021;22(6):572-575.  2.  Abushouk AI, Yunusa I, Elmehrath AO, et al. Quality assessment of published systematic  reviews in high impact cardiology journals: revisiting the evidence pyramid. Frontiers in  Cardiovascular Medicine 2021;8:671569.  3.  Borah R, Brown AW, Capers PL, Kaiser KA. Analysis of the time and workers needed to condu systematic reviews of medical interventions using data from the PROSPERO registry. BMJ ope 2017;7(2):e012545.  4.  Munn Z, Peters MD, Stern C, Tufanaru C, McArthur A, Aromataris E. Systematic review or  scoping review? Guidance for authors when choosing between a systematic or scoping review  approach. BMC medical research methodology 2018;18:1-7.  5.  Kellermeyer L, Harnke B, Knight S. Covidence and rayyan. Journal of the Medical Library  Association: JMLA 2018;106(4):580.  6.  Chan JL, Murphy KA, Sarna JR. Myoclonus and cerebellar ataxia associated with COVID-19: a case report and systematic review. Journal of Neurology 2021:1-32.  7.  (https://www.covidence.org/blog/machine-learning-the-game-changer-for-trustworthy-evidence 8.  Gartlehner G, Affengruber L, Titscher V, et al. Single-reviewer abstract screening missed 13  percent of relevant studies: a crowd-based, randomized controlled trial. Journal of clinical  epidemiology 2020;121:20-28.  9.  Marshall IJ, Wallace BC. Toward systematic review automation: a practical guide to using  machine learning tools in research synthesis. Systematic Reviews 2019;8(1):163. DOI:  10.1186/s13643-019-1074-9.  10.  Rasheed Z, Waseem M, Systä K, Abrahamsson P. Large language model evaluation via multi a agents: Preliminary results. arXiv preprint arXiv:240401023 2024. 
11.  Wang S, Scells H, Zhuang S, Potthast M, Koopman B, Zuccon G. Zero-shot Generative Large  Language Models for Systematic Review Screening Automation.  European Conference on  Information Retrieval: Springer; 2024:403-420.  12.  Zaki M, Namireddy SR, Pittie T, et al. Natural language processing-guided meta-analysis and  structure factor database extraction from glass literature. Journal of Non-Crystalline Solids: X  2022;15:100103. DOI: https://doi.org/10.1016/j.nocx.2022.100103.  13.  National Institute for Health and Care Excellence. Use of AI in evidence generation: NICE  position statement.  (https://www.nice.org.uk/about/what-we-do/our-research-work/use-of-ai-inevidence-generation--nice-positionstatement?utm_medium=social&utm_source=linkedin&utm_campaign=aiposition).  14.  Moher D, Shamseer L, Clarke M, et al. Preferred reporting items for systematic review and meta analysis protocols (PRISMA-P) 2015 statement. Systematic reviews 2015;4:1-9.  15.  Aromataris E, Fernandez R, Godfrey CM, Holly C, Khalil H, Tungpunkom P. Summarizing  systematic reviews: methodological development, conduct and reporting of an umbrella review  approach. JBI Evidence Implementation 2015;13(3):132-140.  16.  Scherbakov D. Large language models in scoping and systematic reviews automation: an  automated systematic review [protocol registration]. DOI:  https://doi.org/10.17605/OSF.IO/EJKSY.  17.  QS World University Rankings 2024.  (https://www.kaggle.com/datasets/joebeachcapital/qsworld-university-rankings-2024?resource=download).  18.  Dinga J. Updated List of Journal Impact Factor 2022 _ Journal Citation Report 2022 and Journal Quartiles 20222022.  19.  Haddaway NR, Westgate MJ. Predicting the time needed for environmental systematic reviews  and systematic maps. Conservation Biology 2019;33(2):434-443.  20.  Liu Z. ChatGPT - A New Milestone in the Field of Education. Applied and Computational  Engineering 2024;35(1):129-133. DOI: 10.54254/2755-2721/35/20230380. 
11.  Wang S, Scells H, Zhuang S, Potthast M, Koopman B, Zuccon G. Zero-shot Generative Large  Language Models for Systematic Review Screening Automation.  European Conference on  Information Retrieval: Springer; 2024:403-420.  12.  Zaki M, Namireddy SR, Pittie T, et al. Natural language processing-guided meta-analysis and  structure factor database extraction from glass literature. Journal of Non-Crystalline Solids: X  2022;15:100103. DOI: https://doi.org/10.1016/j.nocx.2022.100103.  13.  National Institute for Health and Care Excellence. Use of AI in evidence generation: NICE  position statement.  (https://www.nice.org.uk/about/what-we-do/our-research-work/use-of-ai-inevidence-generation--nice-positionstatement?utm_medium=social&utm_source=linkedin&utm_campaign=aiposition).  14.  Moher D, Shamseer L, Clarke M, et al. Preferred reporting items for systematic review and meta analysis protocols (PRISMA-P) 2015 statement. Systematic reviews 2015;4:1-9.  15.  Aromataris E, Fernandez R, Godfrey CM, Holly C, Khalil H, Tungpunkom P. Summarizing  systematic reviews: methodological development, conduct and reporting of an umbrella review  approach. JBI Evidence Implementation 2015;13(3):132-140.  16.  Scherbakov D. Large language models in scoping and systematic reviews automation: an  automated systematic review [protocol registration]. DOI:  https://doi.org/10.17605/OSF.IO/EJKSY.  17.  QS World University Rankings 2024.  (https://www.kaggle.com/datasets/joebeachcapital/qsworld-university-rankings-2024?resource=download).  18.  Dinga J. Updated List of Journal Impact Factor 2022 _ Journal Citation Report 2022 and Journal Quartiles 20222022.  19.  Haddaway NR, Westgate MJ. Predicting the time needed for environmental systematic reviews  and systematic maps. Conservation Biology 2019;33(2):434-443.  20.  Liu Z. ChatGPT - A New Milestone in the Field of Education. Applied and Computational  Engineering 2024;35(1):129-133. DOI: 10.54254/2755-2721/35/20230380. 
21.  Mu Y. The Potential Applications and Challenges of ChatGPT in the Medical Field. International Journal of General Medicine 2024;Volume 17:817-826. DOI: 10.2147/ijgm.s456659.  22.  Tlili A, Shehata B, Adarkwah MA, et al. What if the Devil Is My Guardian Angel: ChatGPT as a Case Study of Using Chatbots in Education. Smart Learning Environments 2023;10(1). DOI:  10.1186/s40561-023-00237-x.  23.  Schopow N, Osterhoff G, Baur D. Applications of the Natural Language Processing Tool  ChatGPT in Clinical Practice: Comparative Study and Augmented Systematic Review. JMIR  Med Inform 2023;11:e48933. DOI: 10.2196/48933.  24.  Teperikidis E, Boulmpou A, Potoupni V, Kundu S, Singh B, Papadopoulos C. Does the long-term administration of proton pump inhibitors increase the risk of adverse cardiovascular outcomes? A ChatGPT powered umbrella review. Acta Cardiol 2023;78(9):980-988. DOI:  10.1080/00015385.2023.2231299.  25.  Noe-Steinmuller N, Scherbakov D, Zhuravlyova A, Wager TD, Goldstein P, Tesarz J. Defining  suffering in pain: a systematic review on pain-related suffering using natural language processing Pain 2024;165(7):1434-1449. DOI: 10.1097/j.pain.0000000000003195.  26.  Agapiou A, Lysandrou V. Interacting with the Artificial Intelligence (AI) Language Model  ChatGPT: A Synopsis of Earth Observation and Remote Sensing in Archaeology. Heritage  2023;6(5):4072-4085. DOI: 10.3390/heritage6050214.  27.  Waffenschmidt S, Knelangen M, Sieben W, Bühn S, Pieper D. Single screening versus  conventional double screening for study selection in systematic reviews: a methodological  systematic review. BMC Med Res Methodol 2019;19(1):132. (In eng). DOI: 10.1186/s12874019-0782-0.  28.  Orenstrakh MS, Karnalim O, Suarez CA, Liut M. Detecting llm-generated text in computing  education: A comparative study for chatgpt cases. arXiv preprint arXiv:230707411 2023.  29.  Liang W, Zhang Y, Wu Z, et al. Mapping the increasing use of llms in scientific papers. arXiv  preprint arXiv:240401268 2024. 
Hosseini M, Resnik DB, Holmes KL. The Ethics of Disclosing the Use of Artificial Intellige Tools in Writing Scholarly Manuscripts. Research Ethics 2023;19(4):449-465. DOI:  10.1177/17470161231180449. 
# Table of Contents
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/55b2/55b2a9c9-60fc-432c-b926-4bd167d47934.png" style="width: 50%;"></div>
# Table S1. LLM Prompts used for screening and extraction. ....................................................................................................................... 25  Table S2. Benchmark of abstract screening phase (N=100 abstracts). ....................................................................................................... 26  Table S3. Benchmark of full-text screening phase (N=30 full-text PDFs). ................................................................................................. 26  Table S4. Benchmark of full-text extraction phase (N=15 full-text PDFs). ................................................................................................ 27 
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/bda4/bda44eef-18d1-4591-a489-7dffcac79b65.png" style="width: 50%;"></div>
Phase of 
the review  
LLM prompt  
Abstract 
screening  
Summarise the text abstract of a full research paper (article), and given the below criteria list, say if the full paper is likely to be included, excluded, or unclear. 
Definition of review. Review is a type of publication that synthesises knowledge from other publications.  
Reviews include systematic, scoping reviews, meta-analysis, evidence synthesic, umbrella and rapid reviews, literature, narrative reviews, and other type of 
reviews.  
A review typically has the following stages:  
research question generation, creating a search strategy for a review, screening of literature, extraction of information, quality and bias assessment, evidence 
synthesis, writing a paper, generating code/plots for the review and generating tables.  
  
Criteria list for exclusion/inclusion.  
Include: Paper should be using some kind of large language models (LLM), like ChatGPT, GPT-3.5, GPT-4, Claude, BERT, BARD, Mistral, PaLM, Gemini, 
Copilot, Llama, Mixtral, and similar.  
Include: Paper should be focused on automation of any stage of the review process listed above.  
Exclude: If any of the Include criteria doesn't match.  
Exclude: The paper is a review itself (types of review are listed above). However, if this review reports that it uses LLM for any review stage (stages of review 
are listed above), then include it.   
Exclude: Paper is not related to automation of any parts of the review.  
Exclude: Paper is a book chapter or compilation of conference papers (but single conference papers should be included).  
Exclude: Paper mentions related technology like code generation with LLM but it is not related to creating a review (see definition of review above).  
Exclude: Abstract and title are too brief and don't contain enough information to make the decision.  
Follow this format:  
1) First provide some explanations why each study should be included or excluded.  
2) Then format your output as follows, strictly follow this format, use equal(=) sign, if study is excluded, write 'answer=excluded', if study is included output 
'answer=included', or if it is unclear write 'answer=unclear'.  
Full-text 
screening  
Look at the research paper (article), and given the below criteria list, say if the full paper is to be included, excluded, or unclear.   
   
Definition of review. Review is a type of publication that synthesises knowledge from other publications.  
Reviews include systematic, scoping reviews, meta-analysis, evidence synthesic, umbrella and rapid reviews, literature, narrative reviews, and other type of 
reviews.  
A review typically has the following stages:  
research question generation, creating a search strategy for a review, review protocol creation, screening of literature, extraction of information, quality and 
bias assessment, evidence synthesis, writing a paper, generating code/plots for the review and generating tables.  
  
Criteria list.  
Paper should be focused on automation on any stage of the review with large language models (LLM).  
If any of the following exclusion reason match, then exclude the article.  
Exclude reason 1: Paper doesn't use some kind of large language models (LLM), like ChatGPT, GPT-3.5, GPT-4, Claude, BERT, BARD, Mistral, PaLM, 
Gemini, Copilot, Llama, Mixtral, and similar.  
Exclude reason 2: Paper doesn't describe automation of any stage of the review process.  
Exclude reason 3: Rather than covering automation of stages of the review process, paper is the review itself. However, if the paper is a review and uses some 
element of review automation to perform the review, then include it.   
Exclude reason 4: Paper matches the focus (review automation with LLM), but it doesn't evaluate or report performance of any phase of the review process. 
Evaluation means verification by human experts. Often after evaluation performance metrics are reported which include, but not limited to: accuracy, F1, 
precision, recall, sensitivity, error rate, time saved, and others.  
Exclude reason 5: Full text couldn't be retrieved.  
 Follow this format:  
1) First provide some explanations why each study should be included or excluded.   
2) Provide citation from text showing what NLP method was used and mental health problem explored.  
3) Output the following:   
include=yes/no/unclear  
exclude_reason=reason_number (choose only one)  
  
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/cb0e/cb0e128d-5b6f-4216-9a37-28e602b37e82.png" style="width: 50%;"></div>
Source: Authors’ own analysis
<div style="text-align: center;">Table S2. Benchmark of abstract screening phase (N=100 abstracts).</div>
  
Sensitivity 
Specificity 
Pos Pred 
Value 
Neg Pred 
Value 
Precision 
Recall 
F1 
Prevalence 
Detection 
Rate 
Detection 
Prevalence 
Balanced 
Accuracy 
Reviewer 1 
vs Consensus 
0.77 
0.98 
0.97 
0.82 
0.97 
0.77 
0.86 
0.48 
0.37 
0.38 
0.88 
Reviewer 2 
vs Consensus 
0.69 
1 
1 
0.78 
1 
0.69 
0.81 
0.48 
0.33 
0.33 
0.84 
Human 
consensus vs 
Consensus 
0.79 
1 
1 
0.84 
1 
0.79 
0.88 
0.48 
0.38 
0.38 
0.9 
LLM vs 
Consensus 
0.94 
0.83 
0.83 
0.93 
0.83 
0.94 
0.88 
0.48 
0.45 
0.54 
0.88 
 Source: Authors’ own analysis 
Source: Authors’ own analysis 
<div style="text-align: center;">Table S3. Benchmark of full-text screening phase (N=30 full-text PDFs)</div>
  
Sensitivity 
Specificity 
Pos Pred 
Value 
Neg Pred 
Value 
Precision 
Recall 
F1 
Prevalence 
Detection 
Rate 
Detection 
Prevalence 
Balanced 
Accuracy 
Reviewer 1 
vs Consensus 
1 
1 
1 
1 
1 
1 
1 
0.76 
0.76 
0.76 
1 
LLM vs 
Consensus 
1 
0.5 
0.86 
1 
0.86 
1 
0.93 
0.76 
0.76 
0.88 
0.75 
 Source: Authors’ own analysis 
Source: Authors’ own analysis 
<div style="text-align: center;">Table S4. Benchmark of full-text extraction phase (N=15 full-text PDFs).</div>
Category 
Country 
Review stage 
automated 
LLM type used 
Performance 
metrics of LLM 
Sample size 
Review type 
automated in the 
study 
Authors opinion 
on LLM 
Citation to 
support authors 
opinion 
Type of funding 
used 
Precision 
1 
0.93 
0.93 
0.8 
0.8 
0.53 
1 
0.93 
0.73 
Recall 
0.86 
0.8 
0.86 
0.33 
0.53 
0.8 
0.93 
0.93 
0.8 
 Source: Authors’ own analysis 
 Source: Authors’ own analysis 
Time-saving and computational costs.  Our review utilized approximately 500$ in OpenAI Azure costs for GPT-4o model.   We estimate that we saved time in screening 3241 abstracts (100 were manually screened for benchmark) by two reviewers with an average rate by a single reviewer of 40 abstracts per hour: 3241*2/40 = 162 hours. In addition, we saved time in screening 270 full-text publications (30 were manually screened for benchmark) by two reviewers with an average rate by a single reviewer of 10 full-texts per hour: 270*2/10 = 54 hours. We saved time in full-text extraction of 157 full-text publications (15 were manually extracted for benchmark) for 2 reviewers, but we had to do manual extraction of all papers for some categories where LLM precision/recall was low spending about 15 minutes per each publication, thus, assuming average rate of a single reviewer at 2 full-texts per hour we saved 157*2/2 – 157/4 = 118 hours. In addition, we saved time on drafting and code generation, with an estimated time saving of 50 hours.   Thus, we estimate total time saving of 334 person-hours. 
Figure S5. LLM model types used in the studies
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d9b0/d9b031f8-6329-4963-b79b-715ab9f7dd78.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure S5. LLM types proposed for automation (models mentioned in 2 or more studies shown). Source: ChatGPT-generated code using extracted data </div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ce0f/ce0f970d-dd95-4825-afdd-b41d5c667407.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure S6. A: Type of citation (a review with LLM usage or a methods paper), B: Overall opinion of citation authors on LLM usage in review, C: Funding sources reported in the study, D: Overall quality of evidence. Source: ChatGPT-generated code using extracted data </div>
<div style="text-align: center;">Table S7. Complete table of extracted categories. † denotes categories that were verified by a human reviewer to ensure precision of extraction. </div>
Study 
Title 
Country/
US State 
Review stage† 
 
Review type† 
LLM type† 
Performance metrics† 
Other metrics 
reported† 
Details on performance 
metrics 
Sample size† 
Time savings 
reported 
Review 
or 
methods 
study† 
Fun-
ding 
Quality of 
evidence  
Overall 
opinion† 
Citation from study 
Guo, 2024 
[1] 
Automated Paper Screening for 
Clinical Reviews Using Large 
Language Models: Data 
Analysis Study 
Canada 
Title and abstract 
screening 
Systematic 
review, Scoping 
review 
GPT / 
ChatGPT 
GPT-4.Title and abstract 
screening.Accuracy=91.0; 
GPT-4.Title and abstract 
screening.F1=60.0 
Yes 
Accuracy: computed by 
dividing papers selected 
by both GPT and human 
reviewers by the total 
number of papers. Macro 
F1-score: not specified in 
detail. Sensitivity: 
calculated for both 
included and excluded 
papers. Interrater 
reliability (kappa and 
PABAK): computed 
against the human-
reviewed papers. 
24307 
Reduction in 
Screening Time 
with Gpt for the 
Noa Dataset Was 
Approximately 
643 Minutes and 
Cost 
Approximately 25 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
High 
Positive 
"Large language models have 
the potential to streamline the 
clinical review process, save 
valuable time and effort for 
researchers, and contribute to 
the overall quality of clinical 
reviews." 
Haltaufde
rheide, 
2024 [2] 
The Ethics of ChatGPT in 
Medicine and Healthcare: A 
Systematic Review on Large 
Language Models (LLMs) 
Germany 
Searching for 
publications 
Rapid Review, 
Systematic review 
GPT / 
ChatGPT 
Not mentioned / Qualitative 
No 
Not extracted/Not 
applicable 
796 
Not extracted/Not 
applicable 
Review 
paper 
Public 
High 
Mixed 
"Ethical examination of LLMs 
in healthcare is still nascent 
and struggles to keep pace 
with rapid technical 
advancements." 
Sun, 2024 
[3] 
How good are large language 
models for automated data 
extraction from randomized 
trials? 
China 
Data extraction 
Systematic review 
ChatPDF, 
Claude 
ChatPDF.Data 
extraction.Kappa =93.0; 
Claude.Data 
extraction.kappa=80.0 
Yes 
Not extracted/Not 
applicable 
49 
Not extracted/Not 
applicable 
Review 
paper 
Public 
High 
Mixed 
"Whilst promising, the 
percentage of correct 
responses is still unsatisfactory 
and therefore substantial 
improvements are needed for 
current AI tools to be adopted 
in research practice." 
Susnjak, 
2023 [4] 
Prisma-dfllm: An extension of 
prisma for systematic literature 
reviews using domain-specific 
finetuned large language 
models 
New 
Zealand 
Searching for 
publications, 
Title and abstract 
screening, Full-
text screening, 
Data extraction, 
Evidence 
synthesis/summar
ization 
Systematic 
review, 
Systematic review 
GPT / 
ChatGPT 
Not mentioned / Qualitative 
No 
Not extracted/Not 
applicable 
Not specified 
Not extracted/Not 
applicable 
Methods 
paper 
Public 
Medium 
Positive 
"The proposed extended 
PRISMA FLLM checklist of 
reporting guidelines provides a 
roadmap for researchers 
seeking to implement this 
approach." 
Susnjak, 
2024 [5] 
Automating research synthesis 
with domain-specific large 
language model fine-tuning 
New 
Zealand 
Evidence 
synthesis/summar
ization, Data 
extraction 
Systematic review 
GPT / 
ChatGPT, 
Mistral 
Not mentioned / Qualitative 
Yes 
not extracted 
SYNTHESIS 
OF 
KNOWLED
GE=4962 
Not Extracted 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
High 
Positive 
"AI technologies can 
effectively streamline SLRs, 
ensuring both efficiency and 
accuracy in information 
retrieval." 
Tang, 
2023 [6] 
Evaluating large language 
models on medical evidence 
summarization 
USA 
Evidence 
synthesis/summar
ization 
Meta-analysis, 
Systematic review 
GPT / 
ChatGPT 
Not mentioned / Qualitative 
Yes 
Performance metrics were 
calculated by comparing 
the generated summaries 
against reference 
summaries using ROUGE-
L, METEOR, and BLEU 
scores, which measure 
overlap and precision of n-
grams. 
GPT-
3.5.synthesis 
of 
knowledge=5
3 
, 
ChatGPT.syn
thesis of 
knowledge=5
3 
Not Reported 
Methods 
paper 
Public 
High 
Negative 
"Our study demonstrates that 
automatic metrics often do not 
strongly correlate with the 
quality of summaries ... LLMs 
could be susceptible to 
generating factually 
inconsistent summaries and 
making overly convincing or 
uncertain statements, leading 
to potential harm due to 
misinformation." 
Tran, 
2024 [7] 
Sensitivity and Specificity of 
Using GPT-3.5 Turbo Models 
for Title and Abstract Screening 
in Systematic Reviews and 
Meta-analyses 
France 
Title and abstract 
screening 
Rapid Review, 
Systematic review 
GPT / 
ChatGPT 
GPT-35 Turbo.Title and 
abstract 
screening.Recall=87.2; 
GPT-35 Turbo.Title and 
abstract 
screening.Specificity=52.2 
No 
Comparing output of GPT-
3.5 models under balanced 
and sensitive rules with 
original decisions from 
authors at title and abstract 
level, with sensitivities and 
specificities calculated 
using continuity corrected 
cell counts. 
22665 
Reducing the 
Number of 
Citations Before 
Manual Screening 
from 2 to 45 4 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
High 
Mixed 
"The GPT-3.5 Turbo model 
may be used as a second 
reviewer for title and abstract 
screening, at the cost of 
additional work to reconcile 
added false positives." 
Blasingam
e, 2024 [8] 
Evaluating a Large Language 
Model’s Ability to Answer 
USA/Tenn
essee 
Evidence 
synthesis/summar
ization 
Other/Non-
specific 
GPT / 
ChatGPT 
Not mentioned / Qualitative 
Yes 
Not extracted/Not 
applicable 
216 
Not extracted/Not 
applicable 
Methods 
paper 
Public 
High 
Positive 
"we envision this being the 
first of a series of 
investigations designed to 
Clinicians’ Requests for 
Evidence Summaries 
further our understanding of 
how current and future 
versions of generative AI can 
be used and integrated into 
medical librarians workflow" 
Yan, 2023 
[9] 
Leveraging Generative AI to 
Prioritize Drug Repurposing 
Candidates: Validating 
Identified Candidates for 
Alzheimer’s Disease in Real-
World Clinical Datasets 
USA/Tenn
essee 
Evidence 
synthesis/summar
ization 
Meta-analysis 
GPT / 
ChatGPT 
Not mentioned / Qualitative 
Yes 
Calculated using Cox 
proportional hazards 
regression models 
comparing the risk of 
Alzheimers disease in 
individuals exposed to a 
drug repurposing 
candidate and propensity 
score-matched individuals 
never exposed to the drug 
GPT-
4.synthesis of 
knowledge=2
0 
Not Reported 
Methods 
paper 
Public 
Medium 
Positive 
"Our findings suggest that 
ChatGPT can generate quality 
hypotheses for drug 
repurposing... With minimal 
costs, ChatGPT has the 
capacity and scalability to 
substantially accelerate the 
review process." 
Li, 2024 
[10] 
Evaluating the Effectiveness of 
Large Language Models in 
Abstract Screening: A 
Comparative Analysis 
USA/Nort
h Carolina 
Title and abstract 
screening 
Meta-analysis, 
Systematic review 
GPT / 
ChatGPT, 
Google PaLM, 
Llama or 
Alpaca, Hybrid 
ChatGPT4.Title and 
abstract 
screening.Accuracy=90.2; 
ChatGPT4.Title and 
abstract 
screening.Recall=89.1; 
ChatGPT4.Title and 
abstract 
screening.Specificity=90.7; 
ChatGPT35.Title and 
abstract 
screening.Accuracy=73.6; 
ChatGPT35.Title and 
abstract 
screening.Recall=74.1; 
ChatGPT4.Title and 
abstract 
screening.Specificity=78.0; 
Google PaLM.Title and 
abstract 
screening.Accuracy=78.6; 
Google PaLM.Title and 
abstract 
screening.Recall=49.9; 
Google PaLM.Title and 
abstract 
screening.Specificity=96.8; 
Meta Llama 2.Title and 
abstract 
screening.Accuracy=74.8; 
Meta Llama 2.Title and 
abstract 
screening.Recall=91.9; 
Meta Llama 2.Title and 
abstract 
screening.Specificity=65.7; 
Hybrid.Title and abstract 
screening.Accuracy=95.5; 
Hybrid.Title and abstract 
screening.Recall=53.9; 
Hybrid.Title and abstract 
screening.Specificity=98.4 
No 
Sensitivity is defined as 
the number of true 
positives divided by the 
sum of true positives and 
false negatives, specificity 
as the number of true 
negatives divided by the 
sum of true negatives and 
false positives, and 
accuracy as sum of true 
positives and true 
negatives divided by the 
total number of abstracts. 
200 
Processing 200 
Abstracts with 
Each Llm Took 
Approximately 10 
20 Minutes using a 
Single Thread 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
Medium 
Mixed 
"While LLM tools are not yet 
ready to completely replace 
human experts in abstract 
screening, they show great 
promise in revolutionizing the 
process." 
Wilkins, 
2023 [11] 
Automated title and abstract 
screening for scoping reviews 
using the GPT-4 Large 
Language Model 
Australia 
Title and abstract 
screening 
Scoping review 
GPT / 
ChatGPT 
GPT-4.Title and abstract 
screening.Accuracy=84.0; 
GPT-4.Title and abstract 
screening.Recall=71.0; 
GPT-4.Title and abstract 
screening.Specificity=89.0 
Yes 
Accuracy was calculated 
as the proportion of correct 
decisions (both inclusions 
and exclusions) made by 
GPT-4 compared to the 
consensus human reviewer 
decision. Sensitivity was 
calculated as the 
proportion of true 
positives (correct 
inclusions) out of all actual 
positives (sources that 
should be included). 
Specificity was calculated 
GPT-
4.abstract 
screening=11
47 
Not Reported 
Methods 
paper 
Public 
High 
Positive 
"GPTscreenR demonstrates 
the potential for LLMs to 
support scholarly work and 
provides a user-friendly 
software framework that can 
be integrated into existing 
review pipelines." 
Clinicians’ Requests for 
Evidence Summaries 
Yan, 2023 
[9] 
Leveraging Generative AI to 
Prioritize Drug Repurposing 
Candidates: Validating 
Identified Candidates for 
Alzheimer’s Disease in Real-
World Clinical Datasets 
USA/Tenn
essee 
Evidence 
synthesis/summar
ization 
Meta-analysis 
GPT / 
ChatGPT 
Not mentioned / Qualitative 
Yes 
Calculated using Cox 
proportional hazards 
regression models 
comparing the risk of 
Alzheimers disease in 
individuals exposed to a 
drug repurposing 
candidate and propensity 
score-matched individuals 
never exposed to the drug 
GPT-
4.synthesis of 
knowledge=2
0 
Not Repo
Li, 2024 
[10] 
Evaluating the Effectiveness of 
Large Language Models in 
Abstract Screening: A 
Comparative Analysis 
USA/Nort
h Carolina 
Title and abstract 
screening 
Meta-analysis, 
Systematic review 
GPT / 
ChatGPT, 
Google PaLM, 
Llama or 
Alpaca, Hybrid 
ChatGPT4.Title and 
abstract 
screening.Accuracy=90.2; 
ChatGPT4.Title and 
abstract 
screening.Recall=89.1; 
ChatGPT4.Title and 
abstract 
screening.Specificity=90.7; 
ChatGPT35.Title and 
abstract 
screening.Accuracy=73.6; 
ChatGPT35.Title and 
abstract 
screening.Recall=74.1; 
ChatGPT4.Title and 
abstract 
screening.Specificity=78.0; 
Google PaLM.Title and 
abstract 
screening.Accuracy=78.6; 
Google PaLM.Title and 
abstract 
screening.Recall=49.9; 
Google PaLM.Title and 
abstract 
screening.Specificity=96.8; 
Meta Llama 2.Title and 
abstract 
screening.Accuracy=74.8; 
Meta Llama 2.Title and 
abstract 
screening.Recall=91.9; 
Meta Llama 2.Title and 
abstract 
screening.Specificity=65.7; 
Hybrid.Title and abstract 
screening.Accuracy=95.5; 
Hybrid.Title and abstract 
screening.Recall=53.9; 
Hybrid.Title and abstract 
screening.Specificity=98.4 
No 
Sensitivity is defined as 
the number of true 
positives divided by the 
sum of true positives and 
false negatives, specificity 
as the number of true 
negatives divided by the 
sum of true negatives and 
false positives, and 
accuracy as sum of true 
positives and true 
negatives divided by the 
total number of abstracts. 
200 
Processing
Abstracts 
Each Llm 
Approximat
20 Minutes 
Single Th
Wilkins, 
2023 [11] 
Automated title and abstract 
screening for scoping reviews 
using the GPT-4 Large 
Language Model 
Australia 
Title and abstract 
screening 
Scoping review 
GPT / 
ChatGPT 
GPT-4.Title and abstract 
screening.Accuracy=84.0; 
GPT-4.Title and abstract 
screening.Recall=71.0; 
GPT-4.Title and abstract 
screening.Specificity=89.0 
Yes 
Accuracy was calculated 
as the proportion of correct 
decisions (both inclusions 
and exclusions) made by 
GPT-4 compared to the 
consensus human reviewer 
decision. Sensitivity was 
calculated as the 
proportion of true 
positives (correct 
inclusions) out of all actual 
positives (sources that 
should be included). 
Specificity was calculated 
GPT-
4.abstract 
screening=11
47 
Not Repo
as the proportion of true 
negatives (correct 
exclusions) out of all 
actual negatives (sources 
that should be excluded). 
Oami, 
2024 [12] 
Accuracy and reliability of data 
extraction for systematic 
reviews using large language 
models: A protocol for a 
prospective study 
Japan 
Data extraction 
Systematic review 
GPT / 
ChatGPT, 
Claude, Google 
Bard / Gemini 
Not mentioned / Qualitative 
Yes 
Accuracy, F1, Precision, 
and Recall were calculated 
by comparing LLM-
extracted data to a 
reference standard created 
by human reviewers. 
Not 
extracted/Not 
applicable 
Substantial 
Reduction in Time 
Compared to 
Conventional 
Methods Exact 
Time Savings not 
Reported 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
High 
Mixed 
"T
and
r
p
ext
ac
Woelfle, 
2024 [13] 
Benchmarking Human-AI 
Collaboration for Common 
Evidence Appraisal Tools 
Switzerlan
d, 
USA/Calif
ornia 
Quality and bias 
assessment 
Meta-analysis, 
Systematic review 
Claude, GPT / 
ChatGPT, 
Mistral 
Claude-3-Opus.Quality and 
bias 
assessment.Accuracy=70.0; 
Claude-2.Quality and bias 
assessment.Accuracy=70.0; 
GPT-4.Quality and bias 
assessment.Accuracy=69.0; 
GPT-35.Quality and bias 
assessment.Accuracy=63.0; 
Mixtral-8x22B.Quality and 
bias 
assessment.Accuracy=64.0; 
Claude-3-Opus.Quality and 
bias 
assessment.Accuracy=74.0; 
Claude-2.Quality and bias 
assessment.Accuracy=63.0; 
GPT-4.Quality and bias 
assessment.Accuracy=70.0; 
GPT-35.Quality and bias 
assessment.Accuracy=53.0; 
Mixtral-8x22B.Quality and 
bias 
assessment.Accuracy=59.0; 
Claude-3-Opus.Quality and 
bias 
assessment.Accuracy=45.0; 
Claude-2.Quality and bias 
assessment.Accuracy=44.0; 
GPT-4.Quality and bias 
assessment.Accuracy=38.0; 
GPT-35.Quality and bias 
assessment.Accuracy=55.0; 
Mixtral-8x22B.Quality and 
bias 
assessment.Accuracy=48.0 
Yes 
Agreement with human 
consensus measured by 
accuracy (agreement 
fraction) and Cohens 
kappa. 
Claude-3-
Opus.bias or 
quality 
assessment=5
04, Claude-
2.bias or 
quality 
assessment=5
04, GPT-
4.bias or 
quality 
assessment=5
04, GPT-
3.5.bias or 
quality 
assessment=5
04, Mixtral-
8x22B.bias 
or quality 
assessment=5
04 
 
Claude-3-
Opus.bias or 
quality 
assessment=1
12, Claude-
2.bias or 
quality 
assessment=1
12, GPT-
4.bias or 
quality 
assessment=1
12, GPT-
3.5.bias or 
quality 
assessment=1
12, Mixtral-
8x22B.bias 
or quality 
assessment=1
12 
 
Claude-3-
Opus.bias or 
quality 
assessment=5
6, Claude-
2.bias or 
quality 
assessment=5
6, GPT-
4.bias or 
quality 
assessment=5
6, GPT-
3.5.bias or 
quality 
Not Reported 
Methods 
paper 
Public 
High 
Mixed 
app
c
hum
of
as the proportion of true 
negatives (correct 
exclusions) out of all 
actual negatives (sources 
that should be excluded). 
Accuracy, F1, Precision, 
and Recall were calculated 
by comparing LLM-
extracted data to a 
reference standard created 
by human reviewers. 
Not 
extracted/Not 
applicable 
Substantial 
Reduction in Time 
Compared to 
Conventional 
Methods Exact 
Time Savings not 
Reported 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
High 
Mixed 
"This study aims to explore 
and evaluate the effectiveness 
of LLMs in systematic 
reviews, focusing on their 
potential to automate data 
extraction while ensuring high 
accuracy and minimal bias." 
Agreement with human 
consensus measured by 
accuracy (agreement 
fraction) and Cohens 
kappa. 
Claude-3-
Opus.bias or 
quality 
assessment=5
04, Claude-
2.bias or 
quality 
assessment=5
04, GPT-
4.bias or 
quality 
assessment=5
04, GPT-
3.5.bias or 
quality 
assessment=5
04, Mixtral-
8x22B.bias 
or quality 
assessment=5
04 
 
Claude-3-
Opus.bias or 
quality 
assessment=1
12, Claude-
2.bias or 
quality 
assessment=1
12, GPT-
4.bias or 
quality 
assessment=1
12, GPT-
3.5.bias or 
quality 
assessment=1
12, Mixtral-
8x22B.bias 
or quality 
assessment=1
12 
 
Claude-3-
Opus.bias or 
quality 
assessment=5
6, Claude-
2.bias or 
quality 
assessment=5
6, GPT-
4.bias or 
quality 
assessment=5
6, GPT-
3.5.bias or 
quality 
Not Reported 
Methods 
paper 
Public 
High 
Mixed 
"Current LLMs alone 
appraised evidence worse than 
humans. Human-AI 
collaboration may reduce 
workload for the second 
human rater for the assessment 
of reporting (PRISMA) and 
methodological rigor 
(AMSTAR) but not for 
complex tasks such as 
PRECIS-2." 
assessment=5
6, Mixtral-
8x22B.bias 
or quality 
assessment=5
6 
Schmidt, 
2024 [14] 
Exploring the use of a Large 
Language Model for data 
extraction in systematic 
reviews: a rapid feasibility 
study 
United 
Kingdom 
Data extraction 
Systematic review 
GPT / 
ChatGPT 
GPT-4.Data 
extraction.Accuracy=80.0 
No 
Each of the models 
responses was rated either 
complete, partial, or 
incorrect by two 
reviewers. If the models 
response contained all 
essential information or 
correctly did not provide a 
response when information 
was absent, it was rated 
complete. If some relevant 
information was present 
but missing other essential 
information, it was rated 
partial. Entirely incorrect 
or misleading responses 
were rated incorrect. 
100 
Not Reported 
Methods 
paper 
Public 
Medium 
Mixed 
"Our results show that there 
might be value in using LLMs, 
for example as second or third 
reviewers. However, caution is 
advised when integrating 
models such as GPT-4 into 
tools." 
Yun, 2024 
[15] 
Automatically Extracting 
Numerical Results from 
Randomized Controlled Trials 
with Large Language Models 
USA/Mass
achusetts 
Data extraction 
Meta-analysis 
GPT / 
ChatGPT, 
Llama or 
Alpaca, 
Mistral, 
Gemma, 
OLMo 
GPT-4.Data 
extraction.F1=73.5; 
GPT-35.Data 
extraction.F1=68.0; 
Alpaca.Data 
extraction.F1=0.0; 
Mistral.Data 
extraction.F1=57.6; 
Gemma.Data 
extraction.F1=59.0; 
OLMo.Data 
extraction.F1=42.4; 
LLaMA.Data 
extraction.F1=12.4; 
BioMistral.Data 
extraction.F1=27.5 
Yes 
Accuracy calculated as the 
proportion of exact 
matches; F1 calculated for 
binary and continuous 
outcomes; MSE calculated 
as the mean standardized 
error of the log odds ratio. 
172 
Not Reported 
Methods 
paper 
Public 
Medium 
Mixed 
"The takeaway from this work 
is that modern LLMs offer a 
promising path toward fully 
automatic meta-analysis, but 
further improvements are 
needed before this will be 
reliable." 
Tsai, 2024 
[16] 
Comparative Analysis of 
Automatic Literature Review 
Using Mistral Large Language 
Model and Human Reviewers 
Taiwan 
Searching for 
publications, 
Title and abstract 
screening, Full-
text screening, 
Data extraction 
Systematic review 
Mistral 
Not mentioned / Qualitative 
Yes 
Not extracted/Not 
applicable 
50 
Time Saving Was 
Reported as 
Mistral Llm 
Completing the 
Review Process in 
17 Hours 
Compared to 100 
Hours by Human 
Reviewers 
Methods 
paper 
Public 
Medium 
Mixed 
"The findings indicate that 
while the Mistral LLM 
significantly surpasses human 
efforts in terms of efficiency 
and scalability, it occasionally 
lacks the analytical depth and 
attention to detail that 
characterize human reviews. 
Despite these limitations, the 
model demonstrates 
considerable potential in 
standardizing preliminary 
literature reviews." 
Robinson, 
2023 [17] 
Bio-SIEVE: Exploring 
Instruction Tuning Large 
Language Models for 
Systematic Review Automation 
United 
Kingdom 
Title and abstract 
screening 
Systematic review 
GPT / 
ChatGPT, 
Llama or 
Alpaca, 
Guanaco 
ChatGPT.Title and abstract 
screening.Accuracy=60.0; 
ChatGPT.Title and abstract 
screening.Precision=59.0; 
ChatGPT.Title and abstract 
screening.Recall=96.0; 
LLaMA.Title and abstract 
screening.Accuracy=74.0; 
LLaMA.Title and abstract 
screening.Precision=82.5; 
LLaMA.Title and abstract 
screening.Recall=71.5; 
Guanaco.Title and abstract 
screening.Accuracy=67.2; 
Guanaco.Title and abstract 
screening.Precision=72.5; 
Guanaco.Title and abstract 
screening.Recall=84.0 
No 
Accuracy, Precision, and 
Recall were calculated 
based on the comparison 
of model predictions to the 
annotated labels in the test 
set. 
ChatGPT.abs
tract 
screening=10
01, 
LLaMA.abstr
act 
screening=10
01, 
Guanaco.abst
ract 
screening=10
01 
Not Reported 
Methods 
paper 
Public 
High 
Positive 
"Bio-SIEVE lays the 
foundation for LLMs 
specialised for the SR process, 
paving the way for future 
developments for generative 
approaches to SR automation." 
Uittenhov
e, 2024 
[18] 
Large Language Models in 
Psychology: Application in the 
Context of a Systematic 
Literature Review. 
Switzerlan
d 
Data extraction 
Systematic review 
GPT / 
ChatGPT 
GPT-4 turbo.Data 
extraction.Accuracy=95.0; 
GPT-4 turbo.Data 
extraction.Recall=96.2; 
GPT-4 turbo.Data 
extraction.Specificity=94.0; 
GPT-4 turbo.Data 
extraction.Accuracy=92.5; 
GPT-4 turbo.Data 
extraction.Recall=96.3; 
GPT-4 turbo.Data 
extraction.Specificity=84.2 
Yes 
Cohens Kappa was 
calculated for inter-rater 
reliability. Sensitivity was 
calculated as TP / (TP + 
FN). Specificity was 
calculated as TN / (TN + 
FP). Accuracy was 
calculated as (TP + TN) / 
(TP + TN + FP + FN). The 
Area Under the ROC 
Curve (AUC) was also 
calculated. 
extraction of 
data=39 
articles 
The Llm 
Completed Our 
Coding Tasks 
Significantly 
Faster than the 
Human Coders 
Taking Only a few 
Hours Compared 
to Several Days 
Methods 
paper 
Public 
Medium 
Positive 
"Our results suggest that 
researchers and LLMs can 
work synergistically, 
improving efficiency, cost-
effectiveness, and quality of 
the systematic literature 
review process." 
Wang, 
2024 [19] 
MetaMate: Large Language 
Model to the Rescue of 
Automated Data Extraction for 
Educational Systematic 
Reviews and Meta-analyses 
USA 
Data extraction 
Systematic 
review, Meta-
analysis 
GPT / 
ChatGPT 
GPT-4 turbo.Data 
extraction.Precision=93.8; 
GPT-4 turbo.Data 
extraction.Recall=90.0; 
GPT-4 turbo.Data 
extraction.F1=91.8 
No 
Precision, recall, and F1 
score were calculated 
based on correctly 
extracted data (CED), 
missing data (MD), and 
incorrectly extracted data 
(IED). Precision = CED / 
(CED + IED), Recall = 
CED / (CED + MD), F1 
Score = 2 * (Precision * 
Recall) / (Precision + 
Recall) 
extraction of 
data=32 
Not Reported 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
Medium 
Positive 
"These findings suggest that 
MetaMate could potentially 
replace or assist human coders 
in data extraction tasks, while 
maintaining or improving 
performance." 
Huotala, 
2024 [20] 
The Promise and Challenges of 
Using LLMs to Accelerate the 
Screening Process of 
Systematic Reviews 
Canada, 
Finland 
Title and abstract 
screening 
Systematic review 
GPT / 
ChatGPT 
GPT-35.Title and abstract 
screening.Precision=65.0; 
GPT-4.Title and abstract 
screening.Precision=50.0; 
GPT-35.Title and abstract 
screening.Recall=17.6; 
GPT-4.Title and abstract 
screening.Recall=41.7 
Yes 
F1 and accuracy were 
calculated using standard 
formulas: F1 = 2 * 
(precision * recall) / 
(precision + recall), and 
accuracy = (true positives 
+ true negatives) / total 
samples 
abstract 
screening=20 
Not Reported 
Methods 
paper 
Public 
Medium 
Mixed 
"Citation: Using LLMs for text 
simplification in the screening 
process does not significantly 
improve human performance. 
Using LLMs to automate title-
abstract screening seems 
promising, but current LLMs 
are not significantly more 
accurate than human 
screeners." 
Yun, 2023 
[21] 
Appraising the Potential Uses 
and Harms of LLMs for 
Medical Systematic Reviews 
Australia, 
China, 
Greece, 
United 
Kingdom, 
USA 
Drafting a 
publication 
Systematic review 
Galactica, 
BioMedLM, 
GPT / 
ChatGPT 
Not mentioned / Qualitative 
No 
Qualitative analysis was 
conducted based on expert 
interviews to evaluate the 
outputs generated by the 
LLMs. 
NA 
Na 
Methods 
paper 
Public 
Medium 
Mixed 
"Participants noted that LLMs 
are inadequate for producing 
medical systematic reviews 
directly given that they do not 
adhere to formal review 
methods and guidelines." 
Prasad, 
2024 [22] 
Towards Development of 
Automated Knowledge Maps 
and Databases for Materials 
Engineering using Large 
Language Models 
India 
Data extraction 
Systematic review 
GPT / 
ChatGPT, 
Google Bard / 
Gemini 
ChatGPT-35 turbo.Data 
extraction.F1=40.0; 
ChatGPT-35 turbo.Data 
extraction.F1=47.9; 
Google Gemini Pro.Data 
extraction.F1=50.0; 
Google Gemini Pro.Data 
extraction.F1=63.0 
Yes 
F1 score was calculated 
using ROUGE metrics 
with the formula: 2 * 
(Precision * Recall) / 
(Precision + Recall). Exact 
Match and Relaxed Match 
were used to compute 
these values. 
extraction of 
data=7 
Not Reported 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
Medium 
Positive 
"Our method offers efficiency 
and comprehension, enabling 
researchers to extract insights 
more effectively." 
Serajeh, 
2024 [23] 
LLMs in HCI Data Work: 
Bridging the Gap Between 
Information Retrieval and 
Responsible Research Practices 
Iran, Italy 
Data extraction 
Other/Non-
specific 
GPT / 
ChatGPT, 
Llama or 
Alpaca 
GPT35.Data 
extraction.Accuracy=58.0; 
LLama2.Data 
extraction.Accuracy=56.0; 
GPT35.Data 
extraction.meanabsoluteerro
r=7.0; 
Llama2.Data 
extraction.meanabsoluteerro
r=7.6 
No 
Not extracted/Not 
applicable 
300 
Not extracted/Not 
applicable 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
High 
Positive 
"This strategy not only 
ensured accuracy but also 
reduced surveillance risk." 
Wang, 
2024 [24] 
Zero-shot Generative Large 
Language Models for 
Systematic Review Screening 
Automation 
Australia, 
Germany 
Title and abstract 
screening 
Systematic review 
GPT / 
ChatGPT, 
Llama or 
Alpaca 
ChatGPT.Title and abstract 
screening.Recall=87.0; 
ChatGPT.Title and abstract 
screening.Recall=93.0; 
Llama.Title and abstract 
screening.Recall=89.0; 
Llama.Title and abstract 
screening.Recall=97.0; 
Alpaca.Title and abstract 
screening.Recall=91.0; 
Alpaca.Title and abstract 
screening.Recall=99.0 
Yes 
Various performance 
metrics (B-AC, success 
rate, WSS) were computed 
across different datasets by 
comparing the predicted 
inclusion/exclusion against 
the ground truth labels. 
abstract 
screening=60
0000 
Significant 
Screening Time 
Saved Compared 
to State of the Art 
Approaches 
Specific Time 
Savings not 
Quantified 
Methods 
paper 
Public 
High 
Positive 
"Our comprehensive 
evaluation using five standard 
test collections shows that 
instruction fine-tuning plays 
an important role in screening, 
that calibration renders LLMs 
practical for achieving a 
targeted recall, and that 
combining both with an 
ensemble of zero-shot models 
saves significant screening 
time compared to state-of-the-
art approaches." 
Cai, 2023 
[25] 
Utilizing ChatGPT to select 
literature for meta-analysis 
shows workload reduction 
while maintaining a similar 
recall level as manual curation 
The 
Netherland
s 
Title and abstract 
screening 
Meta-analysis 
GPT / 
ChatGPT 
GPT35.screeningtitleandabs
tract.Precision=91.0; 
GPT4.screeningtitleandabst
ract.Precision=94.0; 
gpt4.screeningtitleandabstra
ct.Recall=98.0; 
gpt35.screeningtitleandabstr
act.Recall=96.0; 
gpt35.screeningtitleandabstr
act.F1=94.0; 
gpt4.screeningtitleandabstra
ct.F1=96.0 
No 
Not extracted/Not 
applicable 
1000+ 
Not extracted/Not 
applicable 
Methods 
paper 
Public 
High 
Positive 
"We show here that its 
possible to have automatic 
selection of records for meta-
analysis with ChatGPT by 
developing a pipeline named 
LARS" 
Tao, 2024 
[26] 
GPT-4 Performance on 
Querying Scientific 
Publications: Reproducibility, 
Accuracy, and Impact of an 
Instruction Sheet 
USA/Calif
ornia 
Data extraction 
Systematic review 
GPT / 
ChatGPT 
GPT-4.Data 
extraction.Accuracy=87.0; 
GPT-4.Data 
extraction.Recall=72.0; 
GPT-4.Data 
extraction.Precision=87.0 
No 
Accuracy was defined as 
concordance between the 
correct answer and the 
GPT-4 response for 
Boolean and numerical 
questions. Recall was 
calculated as the 
proportion of true 
positives out of the sum of 
true positives and false 
negatives. Precision was 
calculated as the 
proportion of true 
positives out of the sum of 
true positives and false 
positives. F1 score was the 
harmonic mean of 
precision and recall: 2 x 
(recall * precision) / (recall 
+ precision). 
3600 
The Overall Cost 
of using the Gpt 4 
Api Was 
Significantly 
Reduced to 
Approximately 
Five Fold with the 
Release of Gpt 4 
Turbo which is 
more Cost 
Effective but 
Exact Time 
Savings Were not 
Reported 
Methods 
paper 
Public 
High 
Positive 
"GPT-4 possesses extensive 
knowledge about HIV drug 
resistance and it reproducibly 
answers Boolean, numerical, 
and list questions about HIV 
drug resistance papers. Its 
accuracy, recall, and precision 
of approximately 87%, 73%, 
and 87% without human 
feedback demonstrate its 
potential at performing this 
task." 
Tovar, 
2023 [27] 
AI Literature Review Suite 
USA/Tenn
essee 
Searching for 
publications, 
Data extraction 
Literature/Narrati
ve review 
GPT / 
ChatGPT, 
Llama or 
Alpaca 
Not mentioned / Qualitative 
No 
Not extracted/Not 
applicable 
Not 
extracted/Not 
applicable 
Not extracted/Not 
applicable 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
Low 
Positive 
"AI Literature Review Suite 
stands as a potent ally for 
researchers, enhancing 
efficiency and quality of 
scholarly endeavors while 
promoting accelerated 
innovation and progress" 
Tang, 
2024 [28] 
Large Language Model in 
Medical Information Extraction 
from Titles and Abstracts with 
Prompt Engineering Strategies: 
A Comparative Study of GPT-
3.5 and GPT-4 
China, 
Hong 
Kong SAR 
Data extraction 
Systematic review 
GPT / 
ChatGPT 
GPT-4.Data 
extraction.Accuracy=68.8; 
GPT-4.Data 
extraction.Accuracy=96.4; 
GPT-35.Data 
extraction.Accuracy=56.8; 
GPT-35.Data 
extraction.Accuracy=99.2 
No 
Comparison of model 
outputs with ground truth 
using BERTScore, 
ROUGE-1, and a self-
developed GPT-4 
evaluator 
100 
8 to 10 Hours of 
Human Labor 
Reduced to under 
5 Minutes Gpt 3 5 
or 40 Minutes Gpt 
4 
Methods 
paper 
Unknow
n/unrep
orted 
sources 
High 
Positive 
"Our result confirms the 
effectiveness of LLMs in 
extracting medical 
information, suggesting their 
potential as efficient tools for 
literature review