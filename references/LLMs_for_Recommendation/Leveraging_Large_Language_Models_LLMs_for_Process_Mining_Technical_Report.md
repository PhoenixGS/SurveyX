# Leveraging Large Language Models (LLMs) for Process Mining (Technical Report)
Alessandro Berti1,2* and Mahnaz Sadat Qafari
1*Process and Data Science Group, RWTH Aachen University, Ahornstrasse 55, Aachen, 52074, NRW, Germany. 2*Center for Process Intelligence, Fraunhofer FIT, Schloss Birlinghoven, Konrad-Adenauer-Straße, Sankt Augustin, 53757, NRW, Germany.
# *Corresponding author(s). E-mail(s): a.berti@pads.rwth-aachen.de; Contributing authors: sadatghafari@gmail.com;
Abstract
This technical report describes the intersection of process mining and large language models (LLMs), specifically focusing on the abstraction of traditional and object-centric process mining artifacts into textual format. We introduce and explore various prompting strategies: direct answering, where the large language model directly addresses user queries; multi-prompt answering, which allows the model to incrementally build on the knowledge obtained through a series of prompts; and the generation of database queries, facilitating the validation of hypotheses against the original event log. Our assessment considers two large language models, GPT-4 and Google’s Bard, under various contextual scenarios across all prompting strategies. Results indicate that these models exhibit a robust understanding of key process mining abstractions, with notable proficiency in interpreting both declarative and procedural process models. In addition, we find that both models demonstrate strong performance in the object-centric setting, which could significantly propel the advancement of the object-centric process mining discipline. Additionally, these models display a noteworthy capacity to evaluate various concepts of fairness in process mining. This opens the door to more rapid and efficient assessments of the fairness of process mining event logs, which has significant implications for the field.
The integration of these large language models into process mining applications may open new avenues for exploration, innovation, and insight generation in the field. Keywords: Large Language Models, Process Mining, GPT-4, Querying Language
The integration of these large language models into process mining applications may open new avenues for exploration, innovation, and insight generation in the field.
Keywords: Large Language Models, Process Mining, GPT-4, Querying Language
# 1 Preliminaries
The following subsections delve into some key preliminaries to provide the necessary background for our discussion. This encompasses the nature of traditional and object-centric event logs (Section 1.1), the distinctions between procedural and declarative process models (Section 1.2), the techniques for extracting numerical features from event logs (Section 1.4), and the understanding of fairness concepts within the scope of process mining (Section 1.5). These topics form the cornerstone of our exploration into the integration of large language models within process mining.
# 1.1 Traditional and Object-Centric Event Logs
Case ID
Activity
Timestamp
Case1
Activity A
2023-07-13 09:00:00
Case1
Activity B
2023-07-13 09:15:00
Case1
Activity C
2023-07-13 09:30:00
Case2
Activity A
2023-07-13 10:00:00
Case2
Activity B
2023-07-13 10:30:00
Case3
Activity A
2023-07-13 11:00:00
Case3
Activity C
2023-07-13 11:30:00
Case3
Activity D
2023-07-13 11:45:00
Event logs are pivotal to process mining, representing empirical data that drives operations such as process discovery, conformance checking, and model enhancement. Extracted from information systems, these logs comprise collections of events, each characterized by an activity and timestamp and associated with a particular process. As primary artifacts, event logs provide a robust basis for data-driven process analysis. Traditional event logs adhere to a format where each event is tied to a specific case, signifying a unique execution of the process. Every event is defined by its activity—representing the operation performed, and a timestamp—indicating when the operation took place. The Extensible Event Stream (XES) standard Acampora et al (2017) has been established for storing and exchanging these event logs, facilitating systematic documentation and analysis of process events. An example of traditional event logs is reported in
Table 1. We have three cases. For Case1, we have three events with activities A,B,C. For Case2, we have two events with activities A and B. For Case3, we have three events with activities A,C,D.
Event ID
Activity
Timestamp
Order
OrderItems
Invoices
Event1
Activity A
2023-07-13 09:00:00
Order1
Item1, Item2
Invoice1
Event2
Activity B
2023-07-13 09:15:00
Order1
Item1, Item2
Invoice1
Event3
Activity C
2023-07-13 09:30:00
Order2
Item3
Invoice2
Event4
Activity A
2023-07-13 10:00:00
Order2
Item3
Invoice2
Event5
Activity B
2023-07-13 10:30:00
Order2
Item4
Invoice2
Event6
Activity A
2023-07-13 11:00:00
Order3
Item5, Item6
Invoice3
Event7
Activity C
2023-07-13 11:30:00
Order3
Item5
Invoice3
Event8
Activity D
2023-07-13 11:45:00
Order3
Item6
Invoice3
Event9
Activity A
2023-07-13 12:00:00
Order4
Item7, Item8
Invoice4
Event10
Activity B
2023-07-13 12:15:00
Order4
Item8
Invoice4
Object-centric event logs, on the other hand, cater to more complex scenarios by associating each event with multiple objects of different types. Each event still holds information about its activity and timestamp, but instead of relating to a single process execution, it ties into multiple entities. The OCEL standard Ghahfarokhi et al (2021) has been proposed as a standard for the storage of object-centric event logs, capturing detailed interaction patterns between diverse process entities. An example is reported in Table 2. For example, we see that the event with identifier Event1 is related to one object of type order, two objects of type order item, and one object of type invoice. Object-centric event logs allow for the capture of complex, intertwined, multi-object scenarios, opening avenues for more nuanced process mining tasks. However, the complexity of these logs presents unique challenges for process mining, such as how to structure the data for analysis. A significant stride has been made in addressing these challenges with the proposal by Adams et al. in their 2022 study Adams et al (2022b) to partition object-centric event logs into ”executions”. Their methodology, inspired by traditional case-oriented process mining, breaks down the intricate graph structure of object-centric event logs into manageable ”process executions”, effectively the equivalent of cases in traditional event logs. By extending the case concept to object-centric data, the authors enable a more refined application of process mining tasks to these logs, enabling practitioners to query specific aspects of a process. Furthermore, through the application of graph isomorphism, they are able to identify equivalent process executions with respect to an attribute, thereby discovering object-centric variants. This approach offers a method for handling the complexities inherent in object-centric event logs without resorting to flattening the data, thereby preserving information and structure. As example, the object-centric event log in Table 2 is split in four distinct connected components ( {Order1, Item1, Item2, Invoice1}, {Order2, Item3, Invoice2}, {Order3, Item5, Item6, Invoice3}, {Order4, Item8, Invoice4} ).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/7908/790847e2-c948-4680-b31a-197b83010fa0.png" style="width: 50%;"></div>
<div style="text-align: center;">Frequency Annotation</div>
# Fig. 1: Example directly-follows graph (DFG) computed on the event log described in Table 1.
Fig. 1: Example directly-follows graph (DFG) computed on the event log described in Table 1.
# 1.2 Procedural Process Models
Procedural process models, as a fundamental tool in process mining, systematically depict the order of activities and their interrelations within a process. These models capture the idea that some activities must precede others, clearly indicating the sequence or procedure that needs to be followed to attain a particular outcome. They represent a step-by-step visualization of a process, guiding us from the beginning to the end, hence the name ’procedural.’ In such models, each step is contingent on the output of the preceding one, reinforcing a precise flow of tasks in a process. Now, let’s delve into some common types of procedural process models.
Fig. 2: Example Petri net obtained from the event log described in Table  using the Inductive miner process discovery algorithm Leemans et al (2013).
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/3f9f/3f9ffc02-dc9c-46ef-a802-feaf612b1bb7.png" style="width: 50%;"></div>
<div style="text-align: center;">Performance Annotation</div>
Springer Nature 2021 LATEX template
A primary example of a procedural process model is the directly-follows graph (DFG) van der Aalst (2016). This model encapsulates the succession relationship between activities, where an edge from activity A to activity B signifies that B has directly followed A in the process. DFGs provide an intuitive, visual way to comprehend the sequence of activities in a process and can be an effective starting point for process discovery. An example of a directly-follows graph, annotated with frequency and performance information, is reported in Figure 1.
Fig. 3: Example BPMN model obtained from the event log described in Table 1 using the Inductive miner process discovery algorithm Leemans et al (2013).
<div style="text-align: center;">Fig. 3: Example BPMN model obtained from the event log described in Table 1 using the Inductive miner process discovery algorithm Leemans et al (2013).</div>
In the realm of formal procedural process models, Petri nets have gained wide acceptance in process mining van der Aalst (1998). They offer a graphical and mathematical modelling tool well suited to capture concurrency, synchronization and shared resources. In a Petri net, places (represented as circles) can contain tokens, and transitions (represented as rectangles) can consume and produce these tokens, in line with predefined rules. Different process discovery algorithms have Petri nets as eventual output, including the Alpha van der Aalst et al (2004), the Inductive Leemans et al (2013) and the Heuristics Weijters and Ribeiro (2011) miners. An example of Petri net model, discovered from Table 1 using the Inductive miner, is presented in Figure 2. Another common procedural modelling notation used in business process management is the Business Process Model and Notation (BPMN 2.0 Aagesen and Krogstie (2015)). It is a rich, graphical representation for specifying business processes in a business process model, offering a more extensive set of symbols compared to Petri nets, thereby allowing more precise business communication Kalenkova et al (2017, 2019). However, the expressiveness of BPMN can lead to ambiguity due to the potential for different interpretations of its constructs Dijkman et al (2008); Ouyang et al (2006). An example of BPMN model is represented in Figure 3. We also provide an example of BPMN 2.0 XML1. In the realm of object-centric process mining, procedural process models extend their scope to encapsulate the multiple object types involved in a process. This leads to the creation of two key variants: object-centric directlyfollows graphs and object-centric Petri nets. An object-centric directly-follows
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/064d/064d0e06-4300-4a2f-bacf-63803f74fc3e.png" style="width: 50%;"></div>
<div style="text-align: center;">Frequency Annotation</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/0983/09839f6e-64d3-4df1-a753-60938243a6f9.png" style="width: 50%;"></div>
<div style="text-align: center;">Performance Annotation</div>
graph Berti and van der Aalst (2023) represents a direct evolution from the traditional directly-follows graph, with each object type having its own graph portraying the succession of activities involving that particular object. This collation of graphs offers a nuanced view of the process, maintaining the simplicity of the traditional format. On the other hand, the object-centric Petri net van der Aalst and Berti (2020) extends traditional Petri nets to capture the lifecycle of each individual object type within the process. Each object type is represented by its own Petri net, and their integration forms the objectcentric Petri net. This detailed representation provides a comprehensive view of the process, accounting for the interactions between the multiple object
Springer Nature 2021 LATEX template
types. These models provide a powerful means for analyzing complex, objectcentric processes, enabling an in-depth understanding of the process from the perspective of individual object types and their interaction.
# 1.3 Declarative Process Models
Declarative process models provide a different perspective compared to the traditional procedural process models. While procedural models specify the explicit sequence of activities in a process, declarative models describe the process through a set of constraints or rules that prescribe what cannot happen, leaving a degree of flexibility for what can happen. A common notation used for declarative modeling in the process mining field is the DECLARE framework Maggi et al (2011). It targets the controlflow perspective of a process by specifying constraints on the sequence, choice, and parallelism of activities. For instance, a DECLARE model can express that if activity ’A’ happens, then activity ’B’ must eventually follow, capturing the interdependence between activities.
Source Activity
Target Activity
Average Time
Standard Deviation
Activity A
Activity B
1350.0
636.40
Activity A
Activity C
1800.0
0.00
Activity B
Activity C
900.0
0.00
Activity A
Activity D
2700.0
0.00
Activity C
Activity D
900.0
0.00
Table 3: Temporal profile model Stertz et al (2020) for the traditional event
Table 3: Temporal profile model Stertz et al (2020) for the traditional even log presented in Table 1.
Furthermore, declarative process models extend beyond control-flow constraints to capture other aspects of the process. Temporal declarative models incorporate temporal constraints into the process Stertz et al (2020); Barrientos et al (2023), providing further refinement on when activities should occur. Data-driven declarative models integrate data conditions to determine the valid sequence of activities, thus introducing a context-sensitive dimension to the process model. An example temporal model is presented in Table 3. Overall, declarative process models offer a flexible and holistic approach to process modeling, allowing for a wide range of scenarios and conditions to be incorporated, thereby addressing the complex, dynamic nature of many real-world processes.
# 1.4 Extraction of Numerical Featu
Feature extraction is a fundamental process in transforming raw event log data into a format that machine learning algorithms can interpret and learn from. In the context of process mining, feature extraction plays a significant role in various tasks, including predictive process monitoring, anomaly detection, and process enhancement, among others de Leoni et al (2016).
Springer Nature 2021 LATEX template
# 8
The primary strategy to convert an event log into numeric features is through the use of encoding techniques such as one-hot encoding. In this strategy, each unique activity in the event log is assigned a distinct binary (0 or 1) feature, thus creating a binary vector representation for each case in the log. This enables machine learning algorithms to effectively capture and learn the presence or absence of specific activities within each case. This transformation is crucial since machine learning algorithms primarily work with numeric data. A numeric feature table computed on the traditional event log in Table 1 is presented in Table 4. Apart from the straightforward one-hot encoding, more advanced feature extraction methods can be considered. For instance, there are aggregate features that summarize specific characteristics of a case, like the total number of events, the mean time between events, or the standard deviation of the time between events. Another advanced technique is sequence encoding, where the order of activities is preserved. This is particularly beneficial when the sequence of activities is of paramount importance to the process. Other techniques include using n-grams Zandkarimi et al (2020), which capture sequences of ’n’ consecutive activities, or using methods from natural language processing, such as word embeddings Junior et al (2020), to encode activities and their context within the case.
Case ID
Activity A
Activity B
Activity C
Activity D
Count
Total Duration
Case1
1
1
1
0
3
30
Case2
1
1
0
0
2
30
Case3
1
0
1
1
3
45
In the realm of object-centric event logs, numeric feature extraction expands to accommodate the complexity and multidimensionality of these logs. Features are designed to represent the intricate dynamics involving multiple object types, their interactions, and temporal associations Berti et al (2023a); Adams et al (2022a). A numeric feature table computed on the object-centric event log in Table 2 is presented in Table 5. Activity features from object-centric event logs encapsulate the actions occurring on specific objects or types of objects. For each unique activity, a binary indicator or count variable can be created to signify whether a particular activity has occurred or how often it has taken place on each object. This not only helps encapsulate the inherent processes but also provides a granular understanding of different object types’ behaviors. Inter-object features delve into the relationships between different objects in an object-centric log. When objects of different types consistently appear together, a feature is devised to symbolize this co-occurrence. This might be a binary flag signaling the presence of such a relationship or a count of
the frequency of these interactions, offering insight into patterns of object interdependencies and interplay. Temporal features are instrumental in encapsulating the timing aspects associated with events. For each object type, these can include the timestamp of the first or last event, the mean time between events, or the overall duration of processes involving the object. The temporal features help shed light on the pace and sequence of events across different objects, an essential aspect of understanding the overall process flow.
Order
Activity A
Activity B
Activity C
Activity D
Related Items
Related Invoices
Order1
1
1
0
0
2
1
Order2
1
1
1
0
2
1
Order3
1
0
1
1
2
1
Order4
1
1
0
0
2
1
Table 5: Numeric feature table (for the object type Order) derived from the object-centric event log presented in Table 2.
Finally, data attribute features cater to any additional attributes tied to the events or objects themselves. If an event or object comes with associated data attributes such as cost or resource allocation, these attributes can be incorporated as features. This inclusion ensures any vital context-specific or process-related information is not overlooked during the feature extraction process Qafari and van der Aalst (2021), thereby enhancing the explanatory power of the derived features. By generating suitable features from event logs, we can create informative inputs for machine learning algorithms, enhancing their ability to model and predict complex process behaviors.
# .5 Fairness Concepts in Process Mi
Fairness in process mining is a central theme of increasing importance Qafari and van der Aalst (2019). In essence, it aims to ensure that the algorithms and models developed based on event data do not unfairly favor or disadvantage certain instances or groups. This is crucial because process mining techniques are often used to inform operational decisions and business strategies that can significantly impact people and resources. For example, they may be used to optimize workflows, allocate resources, or make predictive decisions. If these models or algorithms are biased or unfair, the resulting decisions can be unjust, leading to significant negative consequences. Therefore, maintaining fairness in process mining is not just an ethical imperative, but also a necessary condition for building trust and ensuring the overall sustainability of business processes. Fairness in process mining can be thought of as a continuum that spans from the individual to the group level, incorporating notions of procedural and counterfactual fairness along the way Pohl et al (2022).
Springer Nature 2021 LATEX template
At the individual level, fairness centers on ensuring that every case in a process is treated equitably. This could involve examining each instance independently to determine whether the same rules and decisions were applied consistently. For example, in a loan application process, individual fairness would imply that each application is evaluated based on the same set of criteria and not influenced by irrelevant attributes such as the applicant’s race or gender. Progressing towards the group level, fairness becomes about ensuring equitable outcomes across different groups within a process. This could mean analyzing whether certain groups are systematically disadvantaged compared to others. For example, in an employee onboarding process, group fairness would imply that new hires from all backgrounds receive equal access to resources and opportunities for training and development. The notion of procedural fairness emphasizes the importance of the process itself being fair, beyond merely the outcomes it produces. Procedural fairness is concerned with the transparency, consistency, and unbiased nature of the processes. For instance, in a complaint handling process, procedural fairness would mean that all complaints are handled following the same, predefined steps and procedures without any bias. Counterfactual fairness Qafari and van der Aalst (2021) introduces a more complex, hypothetical element into the fairness equation. It considers whether the outcome of a process would have been the same had an irrelevant attribute been different. For example, in an employee promotion process, counterfactual fairness would mean that an employee’s promotion decision would have been the same, irrespective of their gender. One of the key challenges in the development of fairness-oriented techniques in process mining is the scarcity of publicly available event logs that feature fairness concerns. Such data sets are crucial for testing and refining techniques. They not only provide a ground truth against which the effectiveness of methods can be evaluated, but also highlight the real-world complexities and nuances that such techniques need to handle. Currently, the only public collection of simulated event logs that are designed to exhibit fairness issues is provided in Pohl and Berti (2023). These logs and their characteristics are described in detail in the paper Pohl et al (2023). This availability gap underlines the need for greater openness and collaboration in the field, and the importance of creating and sharing more such resources, so as to advance the development and validation of fairness-oriented process mining techniques.
# 2 Approach
In the forthcoming section, we unpack our unique approach to leveraging large language models for process mining tasks. Initially, we focus on developing textual abstractions for process mining artifacts Section 2.1, a vital foundation for enabling the language model’s comprehension of processrelated entities. This step includes particular emphasis on process models
Springer Nature 2021 LATEX template
and event logs Section 2.1.1 Section 2.1.2. Subsequently, we delineate various prompting strategies Section 2.2 to guide the model in problem resolution. These strategies span from direct answering to multi-prompt answering, and even generation of database queries Section 2.2.1 Section 2.2.2 Section 2.2.3. Through an effective synergy between these components, we navigate the nuanced landscape of process mining with large language models, ensuring efficient analysis and meaningful insights.
# 2.1 Textual Abstractions of Process Mining Artifacts
In order to leverage the capabilities of Large Language Models (LLMs) for process mining, we need to convert process mining artifacts into a format that these models can understand and interpret effectively. The initial and crucial step in this transformation process is the textual encoding of these artifacts. The goal of this encoding is to convey the essential information embedded in the artifacts while maintaining the constraints imposed by the context window of the LLM.
<div style="text-align: center;">Listing 1: Textual abstraction of the directly-follows graph presented in Figure 1 (obtained using the method pm4py.llm.abstract dfg).</div>
If I have a process with flow:
Activity A −> Activity B ( frequency = 2 performance = 1350.0 )
Activity C −> Activity D ( frequency = 1 performance = 900.0 )
Activity B −> Activity C ( frequency = 1 performance = 900.0 )
Activity A −> Activity C ( frequency = 1 performance = 1800.0 )
In our research, we have developed distinct textual abstractions for different types of event logs and process models. On the one hand, we address traditional and object-centric event logs, which provide a chronological record of events related to a particular process. On the other hand, we also consider procedural and declarative process models. Procedural models focus on the sequence of activities necessary to accomplish a process, while declarative models describe the rules and constraints that govern the execution of the process. Our textual abstractions offer a balanced summary of these artifacts that preserve their critical properties while remaining accessible to LLMs.
# 2.1.1 Textual Abstractions of Process Models
One of the fundamental process mining artifacts is the directly-follows graph (DFG) introduced in Section 1.2. To translate this information into a form digestible by Large Language Models (LLMs), we need to convert the DFG into a textual abstraction. This conversion, as exemplified in Listing 1, is a careful process that aims to maintain the core information contained in the graph while conforming to the text-based input restrictions of the LLMs. Each arc in
Springer Nature 2021 LATEX template
the DFG is translated into a textual sentence that describes the transition from one activity to another, along with its frequency and performance. Frequency signifies how often a particular sequence of activities occurs in the process, while performance captures the average time it takes to move from one activity to the next. The arcs are listed in order of decreasing frequency to prioritize the most common transitions in the process. If the total length of the textual abstraction threatens to exceed the context window limitation of the LLM, some of the less frequent arcs may be omitted. This ensures that the model is fed with the most representative information about the process within the imposed text length constraint, thus enabling effective query answering and process analysis. In addition to the standard textual encoding of the DFG, it is worth noting that advanced large language models exhibit significant flexibility in understanding various text representations of process flows. They are capable of interpreting and reasoning about different encodings for the same underlying information. For instance, the transition from ”Activity A” to ”Activity B” could be expressed in several ways, such as ”Activity A is followed by Activity B”, ”Activity A →Activity B”, or even as a pair ”(Activity A, Activity B)”. Despite the stylistic differences, the LLMs can recognize these expressions as conveying the same transition in the process. This flexibility makes LLMs robust and adaptable tools for analyzing process mining artifacts represented in different textual formats.
Listing 2: Textual abstraction of the Petri net presented in Figure 2 (obtained using the method pm4py.llm.abstract petri net).
If I have a Petri net:
places: [ p 3, p 4, p 5, sink, source ]
transitions: [ (0309dbf9−52f9−42a5−a73a−79da3a2ea20a, ’Activity B’), (37ede306
−4d81−499a−8dbe−b23b5a31372a, ’Activity C’), (5f5a9dd2−b140−4e88−
ac73−b05c3ab17b63, ’Activity A’), (eba11fc0−143b−4765−adb0−2
b736b348f88, ’Activity D’), (skip 1, None), (skip 2, None), (skip 3, None) ]
arcs: [ (0309dbf9−52f9−42a5−a73a−79da3a2ea20a, ’Activity B’)−>p 4, (37ede306
−4d81−499a−8dbe−b23b5a31372a, ’Activity C’)−>p 5, (5f5a9dd2−b140−4
e88−ac73−b05c3ab17b63, ’Activity A’)−>p 3, (eba11fc0−143b−4765−adb0
−2b736b348f88, ’Activity D’)−>sink, (skip 1, None)−>p 4, (skip 2, None)
−>sink, (skip 3, None)−>sink, p 3−>(0309dbf9−52f9−42a5−a73a−79
da3a2ea20a, ’Activity B’), p 3−>(skip 1, None), p 4−>(37ede306−4d81−499
a−8dbe−b23b5a31372a, ’Activity C’), p 4−>(skip 2, None), p 5−>(eba11fc0
−143b−4765−adb0−2b736b348f88, ’Activity D’), p 5−>(skip 3, None),
source−>(5f5a9dd2−b140−4e88−ac73−b05c3ab17b63, ’Activity A’) ]
initial marking: [’source:1’]
final marking: [’sink:1’]
Springer Nature 2021 LATEX template
a specific label, while arcs serve as the connectors between places and transitions and are depicted accurately. This approach enables a clear and precise representation of the Petri net’s structure, making it easy for large language models to comprehend and reason about. As with directly-follows graphs, it is important to note that there is not a single ”correct” way to textually encode Petri nets. Advanced large language models are capable of understanding a wide range of textual encodings, provided the chosen method clearly communicates the structure and components of the Petri net. Different encodings might express the same Petri net, using various terminologies or formatting conventions to depict places, transitions, and arcs. As long as the encoding is clear and unambiguous, the language model will be able to understand and analyze the underlying Petri net effectively. In addition, it is crucial to note that, unlike with directly-follows graphs, we cannot employ the same approach to addressing the context window limitation when dealing with Petri nets. The nature of Petri nets makes it necessary to abstract the entire model textually. The interconnected structure of places, transitions, and arcs in a Petri net forms a unified whole that cannot be easily segmented or reduced without potentially losing critical information or misrepresenting the process logic. Hence, the complete Petri net, regardless of its size, needs to be translated into text for the large language model to process effectively. This characteristic emphasizes the importance of developing efficient, concise and clear abstraction methods to avoid exceeding the context window of the large language model.
14
When it comes to DECLARE process models (Section 1.3), a declarative process modeling notation, our approach to abstraction changes. These models are expressed as a set of constraints, each representing specific interdependencies and relationships between process activities. Given the non-trivial nature of these constraints, it becomes crucial to adequately explain them to the large language model. A textual representation of a DECLARE model, as shown in Listing 3, translates these constraints into a readable format. However, unlike with procedural models, the form in which the constraints and explanations are presented is not rigid. Advanced large language models are capable of interpreting the constraints, regardless of how they’re expressed, as long as the explanation remains clear and coherent. However, one of the significant challenges when dealing with DECLARE models is the context window limitation. Real-life processes often involve hundreds of constraints, making their textual representations considerably long. To tackle this issue, one possible preprocessing strategy could be to rename the activities using alphabet letters or abbreviations. This approach considerably reduces the length of the textual encoding, ensuring it fits within the context window of the large language model, without losing the integrity of the model’s information.
Listing
4:
Textual
abstraction
of
temporal
profile
process
model
presented
in
Table
3
(obtained
using
the
method
pm4py.llm.abstract temporal profile).
The temporal profile is a model describing the average and the standard deviation
of the times between couples of activities eventually (not only directly)
following each other in at least a process execution (so in a trace <A,B,C,D>
the couples (A,B) (A,C) (A,D) (B,C) (B,D) (C,D) shall be considered). Given
a positive value ZETA, a deviation occurs in a process execution when the
time between two activities is lower than AVG −ZETA ∗STDEV or greater
than AVG + ZETA ∗STDEV. For this process, the model is:
Activity A −> Activity B : AVG: 1350.00 s STD: 636.40 s
Activity A −> Activity C : AVG: 1800.00 s STD: 0.00 s
Activity B −> Activity C : AVG: 900.00 s STD: 0.00 s
Activity A −> Activity D : AVG: 2700.00 s STD: 0.00 s
Activity C −> Activity D : AVG: 900.00 s STD: 0.00 s
In addition to the abstractions we’ve discussed, there are also temporal profile models to consider. A temporal profile model, another type of declarative process model (Section 1.3), is primarily concerned with the timing aspects of a process. The abstraction for such a model, as shown in Listing 4, encapsulates the temporal dependencies between various pairs of activities. Specifically, for each pair of activities that can potentially follow one another in the process, the abstraction provides the average and standard deviation of the time elapsed between them. Just as with other models, advanced large language models are flexible in terms of the representation used for these constraints. As long as the pairs of activities and their corresponding time intervals are clearly stated, the large language model can interpret them, regardless of how they are presented in the abstraction. This versatility makes advanced large language models a potent tool for generating meaningful insights from abstracted temporal profile models.
<div style="text-align: center;">Listing 5: Textual abstraction of the OC-DFG presented in Figure 4 (obtain using the method pm4py.llm.abstract ocel ocdfg).</div>
using the method pm4py.llm.abstract ocel ocdfg).
If I have an object−centric event log with the following directly follows graph (
split between the different object types):
Object type: Order
”Activity A” −> ”Activity B” (frequency (number of events) = 3, frequency (
number of objects) = 3, duration = 1200.00)
”Activity C” −> ”Activity A” (frequency (number of events) = 1, frequency (
number of objects) = 1, duration = 1800.00)
”Activity A” −> ”Activity C” (frequency (number of events) = 1, frequency (
number of objects) = 1, duration = 1800.00)
”Activity C” −> ”Activity D” (frequency (number of events) = 1, frequency (
number of objects) = 1, duration = 900.00)
Object type: Invoices
”Activity A” −> ”Activity B” (frequency (number of events) = 3, frequency (
number of objects) = 3, duration = 1200.00)
”Activity C” −> ”Activity A” (frequency (number of events) = 1, frequency (
number of objects) = 1, duration = 1800.00)
”Activity A” −> ”Activity C” (frequency (number of events) = 1, frequency (
number of objects) = 1, duration = 1800.00)
”Activity C” −> ”Activity D” (frequency (number of events) = 1, frequency (
number of objects) = 1, duration = 900.00)
Object type: OrderItems
”Activity A” −> ”Activity B” (frequency (number of events) = 2, frequency (
number of objects) = 3, duration = 900.00)
”Activity A” −> ”Activity D” (frequency (number of events) = 1, frequency (
number of objects) = 1, duration = 2700.00)
”Activity C” −> ”Activity A” (frequency (number of events) = 1, frequency (
number of objects) = 1, duration = 1800.00)
”Activity A” −> ”Activity C” (frequency (number of events) = 1, frequency (
number of objects) = 1, duration = 1800.00)
When it comes to object-centric event logs, we often utilize object-centric directly-follows graphs (Section 1.2) as part of the abstraction process. An example of such an abstraction is depicted in Listing 5. In this case, each type of object within the log has a corresponding set of arcs in its directly-follows graph. These arcs illustrate the flow and sequence of activities involving that particular object type. However, similar to traditional directly-follows graphs, we must be mindful of the context window limitation. As a result, arcs with fewer occurrences may be omitted from the textual abstraction to maintain a manageable size. Regardless of this omission, the critical insight here is the pattern of activities and interactions concerning each object type. Like the previously discussed abstractions, the way in which these object-specific arcs are represented in the abstraction does not significantly affect how advanced large language models interpret them. As long as the arcs are clearly stated, the large language model can efficiently process and generate useful insights from them. This flexibility underlines the applicability and utility of advanced
large language models in understanding and analyzing complex object-centric artifacts.
# 1.2 Textual Abstractions of Event Log
Process variants provide a crucial view of a traditional event log by giving us a glimpse into the different paths or sequences of activities that cases may follow in a process. They can be textually abstracted and presented to a Large Language Model (LLM), as illustrated in Listing 6. Each variant is represented along with its frequency and average throughput time. This structure enables the LLM to understand both the variant’s prevalence and the time typically required for its completion. Given the frequency-based ordering of the variants, it is possible to handle limitations imposed by the context window. If the number of process variants is large and exceeds the context window size, less frequent variants can be omitted from the textual abstraction. Despite this omission, the abstraction would still retain the most common patterns, which likely capture a substantial portion of the overall process behavior. The ability of advanced LLMs to understand different textual representations is particularly beneficial here. The representation of process variants does not have to adhere to a strict syntax. It could be a sequence of activities separated by symbols or a plain text description.
<div style="text-align: center;">Listing 6: Textual abstraction of the process variants in Table 1 (obtain using the method pm4py.llm.abstract variants).</div>
using the method pm4py.llm.abstract variants).
If I have a process with the following process variants:
Activity A −> Activity C −> Activity D ( frequency = 1 performance = 2700.0
)
Activity A −> Activity B −> Activity C ( frequency = 1 performance = 1800.0
)
Activity A −> Activity B ( frequency = 1 performance = 1800.0 )
When dealing with traditional event logs (Section 1.1), which are essentially organized collections of events grouped into cases, we can utilize textual abstractions to distill the key features of a specific case, as shown in Listing 7. In this type of abstraction, we highlight the key attributes at the case level and provide a detailed list of all events associated with that case. Each event is principally characterized by an activity and a timestamp, but other event-level attributes can be included as necessary, providing a more detailed snapshot of each event. The crucial aspect of this textual abstraction is that it presents all necessary information to evaluate the progression of a specific case. Consequently, it enables the large language model to answer questions that pertain to a specific case in question, such as identifying any potential issues or irregularities within that case. This makes case-level textual abstraction a powerful tool for detailed, case-specific analysis in process mining.
17
Listing 7: Textual abstraction of Case 1 in Table 1 (obtained using the metho pm4py.llm.abstract case). If I have a case with the following (case) attributes: concept:name = Case1 the case contains the following events (the activity of the event is reported first): Activity A ( timestamp = 2023−07−13 09:00:00+00:00 ; @@case index = 0 ; @@index = 0 ; ) Activity B ( timestamp = 2023−07−13 09:15:00+00:00 ; @@case index = 0 ; @@index = 1 ; ) Activity C ( timestamp = 2023−07−13 09:30:00+00:00 ; @@case index = 0 ; @@index = 2 ; )
A traditional event log can also be translated into a matrix of numerical features (Section 1.4), a format often used in machine learning. Each row in this matrix represents a case from the event log, and each column, or feature, provides some quantitative characteristic of that case. The textual abstraction of this type of matrix can be illustrated as in Listing 8. For every feature in the matrix, we provide its support, i.e., the number of cases where this feature is present or significant. Additionally, we supply quantiles of the feature values, giving a more detailed understanding of the distribution of these values. These aspects combined allow us to understand the relevance and descriptive power of each feature. Given the potential breadth of this kind of matrix, it is essential to consider the context window limitation when producing a textual abstraction. As a solution, one could decide to omit features that have a low support. Such a choice would ensure that the most impactful features, those that occur more frequently, are represented in the abstraction, providing a succinct yet representative snapshot of the event log’s numerical features. This strategy makes this type of textual abstraction extremely valuable for a broad understanding of the event log’s characteristics from a numerical perspective.
Listing
8:
Textual
abstraction
of
features
extracted
from
the
event
log
presented
in
Table
3
(obtained
using
the
method
pm4py.llm.abstract log features).
Given the following features:
Value ’Activity A’ for Event Attribute ’concept:name’: number of non−zero values:
3 ; quantiles of the non−zero: {0.0: 1.0, 0.25: 1.0, 0.5: 1.0, 0.75: 1.0, 1.0: 1.0}
Value ’Activity C’ for Event Attribute ’concept:name’: number of non−zero values:
2 ; quantiles of the non−zero: {0.0: 1.0, 0.25: 1.0, 0.5: 1.0, 0.75: 1.0, 1.0: 1.0}
Value ’Activity B’ for Event Attribute ’concept:name’: number of non−zero values:
2 ; quantiles of the non−zero: {0.0: 1.0, 0.25: 1.0, 0.5: 1.0, 0.75: 1.0, 1.0: 1.0}
Value ’Activity D’ for Event Attribute ’concept:name’: number of non−zero values:
1 ; quantiles of the non−zero: {0.0: 1.0, 0.25: 1.0, 0.5: 1.0, 0.75: 1.0, 1.0: 1.0}
Succession ’Activity A’ −> ’Activity B’ for the Values of the Attribute ’concept:
name’: number of non−zero values: 2 ; quantiles of the non−zero: {0.0: 1.0,
0.25: 1.0, 0.5: 1.0, 0.75: 1.0, 1.0: 1.0}
Succession ’Activity C’ −> ’Activity D’ for the Values of the Attribute ’concept:
name’: number of non−zero values: 1 ; quantiles of the non−zero: {0.0: 1.0,
0.25: 1.0, 0.5: 1.0, 0.75: 1.0, 1.0: 1.0}
Succession ’Activity B’ −> ’Activity C’ for the Values of the Attribute ’concept:
name’: number of non−zero values: 1 ; quantiles of the non−zero: {0.0: 1.0,
0.25: 1.0, 0.5: 1.0, 0.75: 1.0, 1.0: 1.0}
Succession ’Activity A’ −> ’Activity C’ for the Values of the Attribute ’concept:
name’: number of non−zero values: 1 ; quantiles of the non−zero: {0.0: 1.0,
0.25: 1.0, 0.5: 1.0, 0.75: 1.0, 1.0: 1.0}
Similar to the abstraction of traditional event logs, object-centric event logs can also be textually abstracted (Section 1.1), as exemplified in Listing 9. In an object-centric event log, each event is characterized not only by an activity and a timestamp, as in traditional event logs, but also by a set of related objects. These objects extend the dimensionality of the event log, providing more nuanced and contextually rich information about each event. The textual abstraction of an object-centric event log essentially reflects this richer structure. Each event is paired with its associated objects, hence providing a thorough picture of the specific process instance that the event belongs to. However, due to the typically large size of object-centric event logs, a direct abstraction of the entire log could easily exceed the context window limitation of the large language model. To address this issue, it is advisable to execute the textual abstraction against a specific process execution (as in Listing 9), which can be a connected component of related objects, rather than the entire log. This approach allows for localized question answering on a given process execution, ensuring that the resulting prompts are meaningful and contextually appropriate while staying within the context window limitation. The ability to abstract and question specific process executions provides an extra level of flexibility when investigating process behaviours. It can help to unveil patterns or anomalies that are localized to specific instances of the process, contributing to a more granular understanding of the process dynamics.
Listing 9: Textual abstraction of the connected component in Table 2 containing {Order1, Item1, Item2, Invoice1} (obtained using the method pm4py.llm.abstract ocel).
As with traditional event logs, object-centric event logs can also be transformed into a matrix of numerical features (Section 1.4), each representing a unique aspect of the process. These features often delve deeper, exploring the lifecycles of individual objects or interactions between them within the event log’s graph structure. The textual abstraction of such features is presented in Listing 10, with each feature detailed by its support and quantiles of its values, mirroring the approach used for traditional logs. However, due to the increased complexity and the larger number of features that object-centric event logs can offer, dealing with the context window limitation of Large Language Models can be more challenging. The primary strategy here is to focus on the most relevant features - those with the highest support. This prioritization helps to manage the amount of information fed into the LLM, ensuring that the most significant features are included, while those with less support can be omitted to fit within the context window. This approach allows for an effective trade-of between information detail and the size constraints imposed by the LLMs.
Springer Nature 2021 LATEX template
# 2.2 Prompting Strategies
This section introduces three prompting strategies for utilizing large language models (LLMs) in the field of process mining: ”direct answering,” ”multiprompt answering,” and ”generation of database queries.” These strategies vary in complexity and are chosen based on the problem at hand. They make use of textual abstractions and questions, and leverage the capabilities of LLMs for hypothesis generation, result interpretation, and even SQL query formulation, providing an interactive and adaptive approach to process mining.
# 2.2.1 Direct Answering
The ”direct answering” strategy is the most straightforward prompting approach used with large language models in the context of process mining. This strategy involves formulating a prompt which combines a textual abstraction of a process mining artifact with a direct question related to it. Example questions might be ”What are the bottlenecks of the process?” (starting from the process variants abstraction) or ”What are the main anomalies?” (starting from the directly-follows graph abstraction). It is important to note that, in this strategy, the output structure of the response is not predetermined. Rather, the large language model will generate a response in the form of free text. Consequently, this means that the user, while freed from the task of analyzing the event log themselves, must have sufficient background knowledge to interpret the answer correctly. Another key factor for success with the direct answering strategy is the formulation of the question itself. It needs to be concise, clear, and well-structured to enable the large language model to provide a meaningful and relevant response. This strategy, therefore, represents a delicate balance between user expertise, question formulation, and the interpretability of the model’s responses.
# 2.2.2 Multi-Prompt Answering
In certain scenarios, more complex process mining tasks require a series of distinct prompts, each involving different abstractions and questions. This approach, known as multi-prompt answering, leverages the iterative nature of inquiry, where the answer to one question informs the formulation of the next. In this way, the user guides the large language model (LLM) through a sequence of steps to ultimately resolve the overarching process mining problem. A classic example where the multi-prompt answering strategy is useful is in assessing fairness in process mining. Initially, a user might use a feature table abstraction to identify attributes that could potentially lead to discrimination. Once the LLM provides this information, the user formulates a subsequent prompt to investigate if the process execution differs for cases belonging to the protected and non-protected groups. The second prompt is contingent on the results from the first one, as it is necessary to distinguish between cases before delving deeper into potential discrepancies in process execution.
Springer Nature 2021 LATEX template
In another example, the user might want to analyze process conformance. An initial prompt could be used to compare the process model to a log abstraction to identify non-conforming activities. Subsequent prompts could then be designed to examine why these non-conformities are occurring, perhaps by examining specific cases or focusing on particular process variants. Multi-prompt answering is a powerful strategy, as it allows users to conduct a step-by-step, deep-dive analysis. However, it requires careful formulation of prompts and the ability to interpret and act on the LLM responses to guide the inquiry effectively. Therefore, it demands more user engagement than the direct answering strategy.
# 2.2.3 Generation of Database Queries
The third strategy is ”generation of database queries”, which is particularly useful for complex process mining problems that are too intricate to be directly abstracted or solved by the large language model (LLM). Questions like ”What is the average throughput time when activity B occurs more than one day after the start of the case?” or ”Measure the correlation between throughput time and rework in the case” fall into this category. In these situations, the LLM can be leveraged to generate SQL queries based on the user’s initial natural language inquiry. These generated queries can then be executed against the event log, enabling complex data analysis without the user having to manually construct the intricate SQL statements. Yet, this strategy’s potential goes beyond translating natural language questions into SQL queries. It can also be used to generate hypotheses about the event log, derived from certain abstractions like process variants or log features. For instance, the user can present the LLM with such an abstraction and ask it to formulate an hypothesis about the event log expressed as a SQL query. Once the SQL query (representing the hypothesis) is executed and the results are obtained, the LLM can then be employed again to evaluate if the original hypothesis holds true. Should the initial hypothesis be invalidated, the LLM is capable of formulating alternative hypotheses, which can then be transformed into SQL queries and tested against the data. This iterative method, which encompasses generating hypotheses, testing them, and re-assessing based on the outcomes, allows for a more in-depth exploration and understanding of the underlying processes. This approach significantly expands the scope of possible investigations that can be conducted using the LLM. Nevertheless, while this method can streamline complex query generation and hypothesis testing, it does call for a specific set of skills on the part of the user. While the LLM can assist in generating complex queries and hypotheses, the responsibility of result interpretation and subsequent steps of analysis falls on the user. Thus, a certain degree of analytical acumen and domain knowledge is necessary for this approach to be fully effective.
Springer Nature 2021 LATEX template
24
# 3 Tool Support
In this section, we delve into the tool support available for conducting process mining tasks with large language models. pm4py Berti et al (2019), short for Process Mining for Python, is a versatile and comprehensive library that supports the entire spectrum of process mining applications, ranging from data loading and pre-processing, to process discovery, conformance checking, and enhancement. This suite of functionality enables practitioners and researchers to extract valuable insights from event logs and use them for optimizing and monitoring real-world processes. pm4py is developed from the Fraunhofer FIT process mining group and is fully documented at the address https://pm4py. fit.fraunhofer.de.
Method
Abstraction Provided
pm4py.llm.abstract dfg()
Provides the DFG abstraction of a tradi-
tional event log
pm4py.llm.abstract variants()
Provides the variants abstraction of a tra-
ditional event log
pm4py.llm.abstract log attributes()
Provides
the
abstraction
of
the
attributes/columns of the event log
pm4py.llm.abstract log features()
Provides the abstraction of the machine
learning features obtained from an event
log
pm4py.llm.abstract case()
Provides the abstraction of a case (collec-
tion of events)
pm4py.llm.abstract ocel()
Provides the abstraction of an object-
centric
event
log
(list
of
events
and
objects)
pm4py.llm.abstract ocel ocdfg()
Provides the abstraction of an object-
centric event log (OC-DFG)
pm4py.llm.abstract ocel features()
Provides the abstraction of an object-
centric event log (features for ML)
pm4py.llm.abstract event stream()
Provides an abstraction of the (last)
events of the stream related to a tradi-
tional event log
pm4py.llm.abstract petri net()
Provides the abstraction of a Petri net
pm4py.llm.abstract log skeleton()
Provides the abstraction of a log skeleton
model
A significant development in version 2.7.5 of pm4py is the integration with Large Language Models (LLM), bolstering the tool’s capacity for analysis and understanding of process data. This integration enables pm4py to offer a range of abstractions for process data and models, as detailed in Table 6. More features will be introduced in pm4py 2.7.6. Each method in this module provides a different view or representation of the process data, facilitating diverse avenues for exploration and analysis. In addition to these abstractions, pm4py 2.7.5 allows for direct interaction with LLM through specific methods such as pm4py.llm.openai query
Method
Functionality
pm4py.llm.openai query()
Executes
a
prompt
against
OpenAI,
returning the response as string
Table 7: Method for interacting directly with the LLM APIs available in
(Table 7). This function enables users to directly execute prompts agains OpenAI’s language models, obtaining textual responses that can be furthe analyzed or used in subsequent processes.
<div style="text-align: center;">Listing 11: Example usage of the pm4py’s LLM integration on traditional process mining objects</div>
process mining objects
import pm4py
log = pm4py.read xes(”tests/input data/roadtraffic100traces.xes”)
q1 = ”””\n What are the root causes of the performance issues in the process?
Please provide only process and data specific considerations,
no general considerations.”””
print(pm4py.llm.abstract variants(log) + q1)
net, im, fm = pm4py.discover petri net inductive(log)
q2 = ”””\n Can you provide suggestions to improve the process model
based on your domain knowledge?”””
print(pm4py.llm.abstract petri net(net, im, fm) + q2)
A sample usage of pm4py’s LLM integration is shown in Listing 11, where the abstract variants method is used to retrieve the variants of an event log. Subsequently, the Petri net discovered from the event log is abstracted using abstract petri net, and both of these abstractions are used to construct prompts for querying an LLM.
A distinct advantage of the pm4py tool is its support for object-centric process mining, a relatively recent advancement in the process mining field
that provides a more detailed view of complex processes. Object-centric process mining expands the traditional case-centric perspective, allowing for the consideration of multiple interacting objects in a process, thus enabling a more nuanced understanding of process dynamics. The LLM integration in pm4py embraces this object-centric perspective, as demonstrated in the example given in Listing 12. The script loads an objectcentric event log (OCEL) and then uses two specific abstraction methods tailored for this type of data. The method abstract ocel ocdfg is used to generate an abstraction of the Object-Centric Directly Follows Graph (OCDFG), a representation that captures the relationships between the different activities and objects in the log. This abstraction is then used to create a query for the LLM, asking for unusual relationships in the process graph. The method abstract ocel features is used to generate an abstraction of the features of a specific type of object (in this example, ”order”) in the OCEL. This abstraction is then used to create another query for the LLM, asking for odd feature values. These examples showcase the potential of using LLM in conjunction with object-centric process mining. The intricate process insights offered by objectcentric logs, coupled with the rich interpretative capabilities of large language models, provide practitioners and researchers with powerful tools to uncover hidden process knowledge and drive informed process decisions.
# 4 Assessment
In this section, we examine the ability of large language models, specifically Bard and GPT-4, to understand, generate, and answer queries related to process mining tasks. The considered models and their unique characteristics are discussed in Section 4.1. We also outline the specific event logs leveraged in this assessment in Section 4.2. We delve into the models’ capabilities in the realm of direct answering to process mining queries in Section 4.3, where we assess their ability to respond directly to various types of prompts. This exploration is then extended in Section 4.4, where we investigate the models’ capacity for providing answers across multiple prompts, encapsulating more complex interactions. Finally, we evaluate the models’ aptitude in generating database queries from user prompts in Section 4.5. This involves assessing the ability of these models to not just understand a prompt, but to translate that understanding into SQL queries, a crucial aspect in the context of data analysis. By comprehensively assessing these aspects, we aim to gauge the practicality and potential of these large language models in process mining tasks, seeking to illuminate areas of strength as well as identify opportunities for further research and improvement.
Event Log
Number of Events
Number of Cases
Number of Variants
Number of Activities
Road Traffic
561470
150370
231
11
BPI Challenge 2020
56437
10500
99
17
Sepsis
15214
1050
846
16
Conformance Checking Challenge 2019
1394
20
20
29
# 4.1 Considered Large Language Models
This section provides an examination of the two advanced LLMs currently leveraged in the field of process mining: GPT-4, developed by OpenAI, and Google Bard, brought forward by Google AI. Both models are the product of extensive training on vast datasets comprised of a diverse range of text and code, and exhibit remarkable capabilities in a variety of tasks. GPT-4 is a large language model (LLM) developed by OpenAI. It is a powerful tool that can be used for a variety of tasks, including text generation, translation, and question answering. GPT-4 is trained on a massive dataset of text and code, which allows it to generate text that is both coherent and grammatically correct. Google Bard is a large language model developed by Google AI. It is similar to GPT-4 in terms of its capabilities, but it is trained on a different dataset of text and code. This difference in training data means that Google Bard has different strengths and weaknesses than GPT-4.
# 4.2 Considered Event Logs
In this study, we focus on several real-world process event logs to explore and evaluate the capabilities of large language models in the domain of process mining. These event logs encapsulate diverse fields, such as road traffic fine management, hospital case management, and business travel expenses, offering a breadth of scenarios to test the adaptability and accuracy of the language model. The event logs employed in this research are described in detail below.
Event Log
Number of Events
Number of Cases
Number of Variants
Number of Activities
hiring log high-xes.gz
63869
10000
386
12
hiring log low-xes.gz
72094
10000
296
12
hiring log medium-xes.gz
69054
10000
382
12
hospital log high-xes.gz
69528
10000
80
10
hospital log low-xes.gz
70037
10000
106
10
hospital log medium-xes.gz
70124
10000
77
10
lending log high-xes.gz
58822
10000
41
12
lending log low-xes.gz
60746
10000
31
12
lending log medium-xes.gz
58668
10000
33
12
renting log high-xes.gz
89972
10000
496
16
renting log low-xes.gz
96440
10000
508
16
renting log medium-xes.gz
105555
10000
610
16
main
Age
Citizenship
German Proficiency
Gender
Religion
Years of Education
Underlying Condition
Private Insuran
ng
X
X
X
X
X
X
pital
X
X
X
X
X
X
ding
X
X
X
X
X
ting
X
X
X
X
X
X
• Road Traffic Fine Management Process2: real-life event log of an information system managing road traffic fines in an Italian municipality. • BPI Challenge 2020 (Domestic Declarations)3: the travel expense process consists of an employee’s submission, approval by the travel administration, then by the budget owner and supervisor, and possibly a director. The process concludes with a trip or a payment request. There are two types of trips: domestic and international. Domestic trips do not need prior approval, allowing employees to ask for cost reimbursement afterward. • Sepsis Cases Event Log4: the dataset is an anonymized real-life event log of sepsis cases from a hospital, where sepsis is a life-threatening condition often resulting from an infection. Each case represents a patient’s journey through the hospital, with events captured by the hospital’s Enterprise Resource Planning (ERP) system. The events cover a range of different activities, and a multitude of data attributes are recorded such as the group responsible for the activity, test results, and checklist information. While event timestamps have been randomized, the time interval between events within a case remains unchanged. • Conformance Checking Challenge 2019 5: This past version of the conformance checking challenge was centered around a process involved in medical training. Specifically, it revolved around how medical students were taught to place a central venous catheter (CVC) using ultrasound guidance. The CVC procedure involves inserting a tube into a large vein, a process critical for administering fluids or medicines to patients, among other applications. For assessment of the multi-prompt answering strategy, we also use the vent logs provided in Pohl and Berti (2023) and described in Pohl et al (2023), ncluding: • Hiring event logs: these logs chronicle a complex recruitment process with various stages of evaluation and decision-making. The applicant’s journey can be as simple as a straightforward rejection or as elaborate as multi-step screening culminating in a job offer, contingent on a host of influencing factors. • Hospital event logs: this dataset portrays a patient’s journey through a hospital system, starting from the initial check-in at the Emergency Room or Family Department. It progresses through phases of medical
examination, diagnosis, and treatment. It’s worth noting that unsuccessful treatments often necessitate repeat examinations and interventions, illustrating the cyclical nature of healthcare delivery. • Lending event logs: these logs detail a loan application process starting from the scheduling of an initial appointment. It traverses several steps including validation of provided information and underwriting, leading to either loan approval or denial. Further requirements, such as the involvement of a co-signer or an assessment of collateral, may be needed. Some cases may experience direct appointment refusal, highlighting the process’s variability based on applicants’ individual financial circumstances. • Renting event logs: these logs present a comprehensive rental process that begins with a property viewing request and can end with a contract termination. Between these endpoints, stages such as preliminary vetting, property viewing, decision-making, and possibly, intensive screening occur. If successful, a rental agreement is initiated, with possible outcomes ranging from eviction due to late payments to voluntary contract termination by tenants. It’s important to note that not all applications advance to the property viewing phase. The degree of discrimination exhibited in the logs varies across different domains, offering researchers an opportunity to explore tangible real-world situations. Table 9 furnishes fundamental statistics for the log collection, and Table 10 elaborates on the sensitive attributes. It’s important to note that the selection of attributes pertaining to fairness can be a subject of debate; hence, we welcome discussions to enhance our grasp of fairness within the realm of process mining. Detailed descriptions of each log’s attributes and process have been provided to facilitate the identification of potential instances of discrimination. The event logs provided also include the ground truth concerning the classification of cases - whether they belong to the protected group or not. This inclusion enables a comparison between actual case classifications, which are based on the sensitive attributes, and the actual ground truth. Therefore, the capacity to evaluate the accuracy of case categorization into protected and non-protected groups is significantly enhanced. In our exploration of object-centric process mining using large language models, we utilized several publicly accessible logs available at https://www. ocel-standard.org. Specifically, two simulated object-centric event logs were chosen for this analysis: • the ’Order Management’ event log, which can be accessed directly at https://www.ocel-standard.org/1.0/running-example.jsonocel.zip. This log contains the object types ’orders’, ’items’, and ’packages’, all of which were retained for our assessment. • the ’Recruiting’ event log, available at https://www.ocel-standard.org/1. 0/recruiting.jsonocel.zip. In this log, we focused on the ’applications’ and ’offers’ object types.
By leveraging these specific logs, we aimed to evaluate how effectively large language models like Bard and GPT-4 could handle complex, object-centric process mining tasks.
# 4.3 Direct Answering
In this section, we conduct an evaluation of the ability of advanced large language models, specifically Bard and GPT-4, to respond effectively to process mining queries, given the required abstraction. Our assessments of Bard and GPT-4 with regards to traditional process mining and event logs are documented in Table 11. In the table, each query is assigned a unique code, and the appropriate acceptance criteria for the responses generated by the large language models are stipulated. Furthermore, each query is examined across four distinct event logs, namely Road Traffic, BPI Challenge 2020, Sepsis, and CCC19. Observably, both Bard and GPT-4 demonstrate commendable performance when dealing with descriptive queries such as TQ1, TQ2, TQ3, and TQ5. Furthermore, these large language models exhibit a fair understanding of procedural and declarative models, demonstrating capabilities in tasks like play-out and conformance checking, albeit with certain constraints. In particular, GPT-4 displayed a satisfactory level of competency across the diverse range of queries, albeit with some restrictions pertaining to medical processes. Therefore, it is suggested that further refinement and training might be beneficial in enhancing the capability of these models in the realm of process mining.
<div style="text-align: center;"></div>
T-4
Bard
GPT-4
Bard
√
√
√
√
√
√
√
√
√
X
√
X
√
√
√
X
√
X
X
X
X
X
X
X
X
X
X
X
X
X
<div style="text-align: center;"> ev nt</div>
ties on object-centric
on Used
Order Managemen
GPT-4
Bard
√
√
√
√
√
√
√
√
√
√
Springer Nature 2021 LATEX template
In Table 12, we evaluate the performance of large language models in tackling object-centric process mining tasks. Considering the relatively uncomplicated nature of the event logs used for this assessment, both models - Bard and GPT-4 - demonstrate notable proficiency across all tasks. To provide a comprehensive understanding of their capabilities, we document a selection of responses generated by these models in Listing 13, Listing 14, Listing 15, and Listing 16. These examples serve as illustrative evidence of the substantial potential these models exhibit in responding effectively to a diverse set of queries within the realm of object-centric process mining.
From the process variants and the frequency and performance statistics provided, there are a few key steps that could be causing performance issues in the
process: −Rejection by ADMINISTRATION or SUPERVISOR: Variants where the declaration is rejected by the Administration or Supervisor tend to have a higher performance number, which indicates more time or resources consumed in the process. This could be due to the time taken to review the declaration, the process to handle rejection, or the time spent to correct and resubmit the declaration. −Re−submission by EMPLOYEE: After a declaration is rejected, it is often re− submitted by the employee. This step will naturally add to the overall time of the process, as the employee will need to correct the issue that caused the rejection and then re−submit the declaration. −Approval by PRE APPROVER, BUDGET OWNER or SUPERVISOR: Variants that include additional approval steps (by a PRE APPROVER, BUDGET OWNER, or SUPERVISOR) also have higher performance numbers, indicating that these steps are adding to the time taken for the process. This could be due to time−consuming review processes or waiting times for approval. −Missing Approval: There are few cases where declarations are rejected due to missing approvals. This adds an additional step and increases the performance of the process. −Multi−step Processes: Variants with multiple steps, such as those that include rejections, resubmissions, and multiple approvals, show much higher performance numbers. This suggests that processes with multiple stages are taking much longer and consuming more resources. Solutions to these issues might include streamlining the approval process, providing better training or resources to reduce the number of rejections, or implementing a more efficient system for handling re−submissions. Identifying and addressing these root causes could lead to significant improvements in the performance of the process.
# 4.4 Multi-Prompt Answering
In our fairness assessment, we employ the multi-prompt strategy, primarily aiming to discern whether GPT-4 and Bard can accurately identify the protected group. For this purpose, SQL queries were generated through the large language model, utilizing prompts that encapsulate the attributes of the log and their corresponding characterization. An exemplar script that generates such prompts is available at https://github.com/pm4py/pm4py-core/ blob/priv/examples/llm/01 1 protected group identification.py. The aforementioned queries for two event logs are delineated in Figure 5. Upon analysis, we observe that GPT-4’s queries display a more encompassing view of the behavior considered as protected. Conversely, Bard’s queries precisely target a specific subset of cases that are presumably exposed to discrimination.
Listing 17: SQL query provided by
GPT-4 to identify the protected
group in the hiring log high.xes.gz
event log.
SELECT ∗FROM dataframe
WHERE ”case:citizen” = ’False’
OR ”case:gender” = ’False’
OR ”case:german speaking” = ’False’
OR ”case:religious” = ’False’;
Listing 18: SQL query provided
by Bard to identify the protected
group in the hiring log high.xes.gz
event log.
SELECT ∗
FROM dataframe
WHERE ”case:citizen” = ’False’
AND ”case:gender” = ’False’
Listing 19: SQL query provided by
GPT-4 to identify the protected
group in the renting log high.xes.gz
event log.
SELECT ∗FROM dataframe
WHERE ”case:citizen” = ’False’
OR ”case:gender” = ’False’
OR ”case:german speaking” = ’False’
OR ”case:married” = ’False’;
Listing 20: SQL query provided
by Bard to identify the protected
group in the renting log high.xes.gz
event log.
SELECT ∗
FROM dataframe
WHERE ”case:gender” = ’False’
AND ”case:german speaking” = ’
False’
Listing 17: SQL query provided by GPT-4 to identify the protected group in the hiring log high.xes.gz
SELECT ∗FROM dataframe
WHERE ”case:citizen” = ’False’
OR ”case:gender” = ’False’
OR ”case:german speaking” = ’False’
OR ”case:religious” = ’False’;
Fig. 5: SQL queries generated by Bard and GPT-4 to identify the protected groups in the considered event logs.
Subsequently, our focus shifts towards those event logs that signify a substantial degree of discrimination. To assess the classification performance on these logs, we employ an exemplar script accessible at https://github.com/pm4py/pm4py-core/blob/priv/examples/llm/01 2 protected group query.py. The resulting classification quality of GPT-4 and Bard is presented in Table 13 and Table 14. While Bard demonstrates high precision, it is hindered by a low recall and true negative rate, suggesting that it may fail to identify a considerable number of cases where potential discrimination might have taken place. On the other hand, GPT-4, despite exhibiting lower precision, boasts a high recall and true negative rate. This highlights GPT-4’s capability to label a larger number of cases as potentially discriminated, hence reducing the occurrence of false negatives. Having established a set of cases as ’protected’ via GPT-4 and Bard, we now shift our focus towards assessing procedural fairness. The objective here is to uncover any process discrepancies that might hint at discrimination. To achieve this, we craft prompts embodying the process variants abstraction of both ’protected’ and ’non-protected’ cases. These prompts are then fed into the large language model with a request to discern the differences. To ensure a fair and effective comparison, the ’protected’ group, as per the ground truth contained within the simulated event logs, is taken into
Listing 18: SQL query provided by Bard to identify the protected group in the hiring log high.xes.gz event log.
Listing 20: SQL query provided by Bard to identify the protected group in the renting log high.xes.gz event log.
<div style="text-align: center;">Table 13: Quality of the classification done by GPT-4 between the ’protected’ and ’non-protected’ groups.</div>
and ’non-protected’ groups.
Event Log
TP
FP
FN
TN
Precision
Recall
True Neg. Rate
hiring log high
1507
606
1512
6375
0.72
0.50
0.80
hospital log high
1105
1089
1417
6389
0.50
0.44
0.82
lending log high
1501
688
1500
6391
0.69
0.50
0.81
renting log high
1068
1056
1914
5962
0.50
0.35
0.75
<div style="text-align: center;">able 14: Quality of the classification done by GPT-4 between the ’protected’ d ’non-protected’ groups. The excellent recall and true negative rate should</div>
be noticed.
Event Log
TP
FP
FN
TN
Precision
Recall
True Neg. Rate
hiring log high
2977
5768
42
1213
0.34
0.98
0.97
hospital log high
2922
6395
88
595
0.31
0.97
0.87
lending log high
2923
4555
78
2444
0.39
0.97
0.96
renting log high
2716
6749
266
269
0.29
0.91
0.50
consideration. This approach is demonstrated in the example script accessible at https://github.com/pm4py/pm4py-core/blob/priv/examples/llm/02 group fairness factors.py.
Listing 21: Procedural differences between the ’protected’ and ’non-protected’ group identified by GPT-4 in the hiring log high.xes.gz event log.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4689/46891cd3-94dd-46ff-857f-c697c3ef6066.png" style="width: 50%;"></div>
Listing 22: Procedural differences between the ’protected’ and ’non-protected’ group identified by Google Bard in the hiring log high.xes.gz event log.
Refer to Listing 21 and Listing 22 for GPT-4 and Bard’s respective responses pertaining to the hiring event log. Similarly, for the renting event log, GPT-4 and Bard’s responses can be found in Listing 23 and Listing 24 respectively.
Listing 23: Procedural differences between the ’protected’ and ’non-protected’ group identified by GPT-4 in the renting log high.xes.gz event log.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/21a4/21a4f57d-8a48-49bc-b4a9-c0efc72963f5.png" style="width: 50%;"></div>
Listing 24: Procedural differences between the ’protected’ and ’non-protected’ group identified by Google Bard in the renting log high.xes.gz event log.
# Listing 24: Procedural differences between the ’protected’ and ’non-protected’ group identified by Google Bard in the renting log high.xes.gz event log.
Both large language models perform admirably, pinpointing differences in the control-flow and time perspectives. These variances in control-flow correlate with the outcome of a case, the presence of rework, and the number of events in the case. That said, GPT-4’s response stands out for being more comprehensive compared to Bard’s.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ec69/ec697f27-b2c9-4035-84ce-da96752a9f6f.png" style="width: 50%;"></div>
<div style="text-align: center;">Prompt executed against both large language mode</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/84b2/84b252af-518e-4f23-95ea-0f9c80d2a890.png" style="width: 50%;"></div>
<div style="text-align: center;">Wrong response from Bard.</div>
Fig. 6: SQL query generation comparison between GPT-4 and Bard for the “What is the average throughput time when activity ’Send Fine’ occurs more than 50 days after the start of the case?” query executed against the Road Traffic event log.
In our research, we are investigating how large language models can formulate process queries in mainstream SQL language. Our primary focus lies on two prominent models - Bard and GPT-4 - and their ability to translate user-provided natural language queries into executable SQL statements. Furthermore, we are interested in how adept these models are at spontaneously generating hypotheses based on an event log. The models are given basic abstractions such as the list of attributes and process variants within the log. For our analysis, we utilize DuckDB SQL language, a choice motivated by its seamless compatibility with the event log structure used by pm4py. In the present state of affairs, it is evident that some amount of domain knowledge is indispensable to ensure correct query formulation. This includes:
Springer Nature 2021 LATEX template
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a104/a104bab6-b6bf-40ed-9fce-6cf418d08666.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9652/96525b55-734c-4c7c-a01e-c0e92012411d.png" style="width: 50%;"></div>
<div style="text-align: center;">Correct response from GPT-4.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/78c1/78c145a6-273d-4059-892e-a91d79945b34.png" style="width: 50%;"></div>
<div style="text-align: center;">Wrong response from Bard.</div>
• Knowledge of SQL, such as the functions needed to calculate timestamps. • An understanding of process mining concepts, such as how to compute a case’s throughput time or how to derive the process variant from a list of activities. • Awareness of the event log’s key attributes, such as the column that identifies the case, the activity, and the timestamp.
• Knowledge of the variable name that the query should target. While these prerequisites may require specialized knowledge, they are instrumental in ensuring the accuracy and relevance of the SQL queries generated by the large language models. In our study, we endeavored to assess the ability of large language models to convert natural language queries into SQL. Two distinct queries were used for this purpose on the Road Traffic event log.: 1. What is the average throughput time when activity ’Send Fine’ occurs more than 50 days after the start of the case? (Refer to Figure 6 for the full query and the corresponding responses from GPT-4 and Bard). 2. I want to check the average rework for cases below and above the average throughput time (Refer to Figure 7 for the full query and the corresponding responses from GPT-4 and Bard). Our analysis reveals that, given sufficient domain knowledge, GPT-4 has an impressive capability to translate natural language queries into accurate SQL statements. Bard, on the other hand, appears to have significant difficulty making sense of the provided information, often failing to generate executable SQL queries. This discrepancy underscores the importance of adequate domain understanding for successful natural language processing tasks in process mining.
Create Fine −> Send Fine −> Insert Fine Notification −> Add penalty −> Send for Credit Collection ( frequency = 36 performance = 53746600.0 ) Create Fine −> Payment ( frequency = 22 performance = 809181.8181818182 )
Can you provide me a DuckDB SQL query. You should use the EPOCH function of DuckDB to get the timestamp from the date. The data is stored in a Pandas dataframe where each row is an event having the provided attributes (so there are no separate table containing the variant). The dataframe is called ”dataframe”. Please consider the following information: the case identifier is called ”case:concept :name”, the activity is stored inside the attribute ”concept:name”, the timestamp is stored inside the attribute ”time:timestamp”, the resource is stored inside the attribute ”org:resource”. There is not a variant column but that can be obtained as concatenation of the activities of a case. There is not a duration column but that can be obtained as difference between the timestamp of the first and the last event.
To explore the potential of large language models in autonomously formulating hypotheses about event data, we posed a specific prompt to both GPT-4 and Bard (refer to Listing 25). This prompt was formulated such that t encouraged the models to generate hypotheses, starting from the knowledge f the attributes and the process variants present in the data. Here are some hypotheses generated by GPT-4: 1. The majority of cases conclude with the payment of fines after the addition of a penalty. This suggests that the imposition of a penalty may be a decisive factor prompting payment. The corresponding SQL query formulated by GPT-4, along with the results derived from its execution, are depicted in Figure 8. 2. The process of appealing to the prefecture does not appear to be a common practice. This indicates that offenders rarely dispute their fines. The provided SQL query for this hypothesis, along with the subsequent results, can be viewed in Figure 9. 3. There may be a correlation between the amount of fines and the frequency of credit collections, with higher fines potentially leading to an increase in credit collection instances. This could be attributed to the difficulty in paying larger fines. The SQL query and results corresponding to this hypothesis are presented in Figure 10. Bard’s generated hypotheses included the following: 1. The most common process variant is ”Create Fine -→Send Fine -→Insert Fine Notification -→Add penalty -→Send for Credit Collection”. 2. The average payment amount for cases that traverse the complete process appears to be higher compared to cases that are concluded prematurely. 3. The longest durations to complete are typically associated with cases that involve an appeal. Both large language models generated valid hypotheses. However, a critical istinction arises from their ability to translate these hypotheses into exeutable SQL queries. While GPT-4 demonstrated this capability effectively, Bard did not manage to formulate corresponding SQL queries, emphasizing he advantage of incorporating explicit domain knowledge into the language model.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/23d5/23d5504d-1300-4198-87da-e505e3db9b85.png" style="width: 50%;"></div>
<div style="text-align: center;">SQL query generated by GPT-4.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/4630/4630d51b-3df8-48c4-ae58-f1c251e316d5.png" style="width: 50%;"></div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/05e4/05e47b3c-8466-42da-86a2-b896f032cfca.png" style="width: 50%;"></div>
<div style="text-align: center;">SQL query generated by GPT-4.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/d55f/d55f49b2-5319-4abc-9e8c-94c2cc240f5b.png" style="width: 50%;"></div>
<div style="text-align: center;">Validity of the hypothesis assessed by GPT-4.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/9a8f/9a8fe11a-883c-4faf-8598-02dcac469ca1.png" style="width: 50%;"></div>
<div style="text-align: center;">SQL query generated by GPT-4.</div>
<div style="text-align: center;">Result obtained after executing the SQL query.</div>
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/fa3b/fa3b85d8-d32c-47a7-8016-7aa53e660976.png" style="width: 50%;"></div>
Springer Nature 2021 LATEX template
# 5 Related Work
The field of process mining is not isolated, but intersects with various domains such as database technologies, natural language processing (NLP), and the use of large language models. Additionally, it relies heavily on domain knowledge for effective task execution. In the subsequent sections, we explore these intersections and dependencies, providing an overview of the state-of-the-art in these domains and their relevance to process mining. We delve into the interaction of database technologies with process mining (Section 5.1), the intersection of process mining and NLP (Section 5.2), and the importance of domain knowledge in process mining tasks (Section 5.3). This is followed by a thorough examination of the current state of large language models (Section 5.4) and their application within the realm of Business Process Management (Section 5.5). This examination helps illuminate the context and motivation behind our study.
# 5.1 Database Technologies for Process Mining
Database technologies play a crucial role in process mining, helping manage the increasing amounts of process-related data. They provide ways to extract and analyze this data, leading to valuable insights. In this section, we delve into several of these technologies and their contributions to process mining. The book Polyvyanyy (2022) considers the recognition and management of core techniques for retrieving and manipulating process-related artifacts. These core techniques are deemed reusable across various contexts and use cases such as process compliance, standardization, and monitoring, among others. By focusing on these reusable techniques, their aim is to increase efficiency and streamline the application of process querying across various projects and disciplines. Another approach Polyvyanyy et al (2017) proposes a framework to guide the creation of process querying methods. This framework serves as a blueprint for managing repositories of executed and designed processes and their relationships. This approach is instrumental in facilitating strategic decision-making, as well as laying the groundwork for the next generation of Business Intelligence platforms. The paper Sch¨onig et al (2016) delve into a more specific solution, introducing a mining approach that works directly on relational event data through conventional SQL queries. By leveraging the power of database technologies, this approach is able to efficiently mine data without limiting the detection of certain control-flow constraints, thus proving to be fast and effective. In Baader and