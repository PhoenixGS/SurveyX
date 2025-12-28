GRAPH-CONSTRAINED REASONING: FAITHFUL REA-
SONING ON KNOWLEDGE GRAPHS WITH LARGE LAN-
GUAGE MODELS
Linhao Luo1∗, Zicheng Zhao2∗, Chen Gong2, Gholamreza Haffari1, Shirui Pan3†
1Monash University 2Nanjing University of Science and Technology 3Griffith University
{Linhao.Luo,Gholamreza.Haffari}@monash.edu
{zicheng.zhao,chen.gong}@njust.edu.cn, s.pan@griffith.edu.au
ABSTRACT
Large language models (LLMs) have demonstrated impressive reasoning abilities,
but they still struggle with faithful reasoning due to knowledge gaps and halluci-
nations. To address these issues, knowledge graphs (KGs) have been utilized to
enhance LLM reasoning through their structured knowledge. However, existing
KG-enhanced methods, either retrieval-based or agent-based, encounter difficul-
ties in accurately retrieving knowledge and efficiently traversing KGs at scale.
In this work, we introduce graph-constrained reasoning (GCR), a novel frame-
work that bridges structured knowledge in KGs with unstructured reasoning in
LLMs. To eliminate hallucinations, GCR ensures faithful KG-grounded reason-
ing by integrating KG structure into the LLM decoding process through KG-Trie,
a trie-based index that encodes KG reasoning paths. KG-Trie constrains the de-
coding process, allowing LLMs to directly reason on graphs and generate faith-
ful reasoning paths grounded in KGs. Additionally, GCR leverages a lightweight
KG-specialized LLM for graph-constrained reasoning alongside a powerful gen-
eral LLM for inductive reasoning over multiple reasoning paths, resulting in ac-
curate reasoning with zero reasoning hallucination. Extensive experiments on
several KGQA benchmarks demonstrate that GCR achieves state-of-the-art per-
formance and exhibits strong zero-shot generalizability to unseen KGs without
additional training. Code is available at https://github.com/RManLuo/
graph-constrained-reasoning.
1
INTRODUCTION
Large language models (LLMs) have shown impressive reasoning abilities in handling complex
tasks (Qiao et al., 2023; Huang & Chang, 2023), marking a significant leap that bridges the gap
between human and machine intelligence. However, LLMs still struggle with conducting faithful
reasoning due to issues of lack of knowledge and hallucination (Huang et al., 2024; Wang et al.,
2023). These issues result in factual errors and flawed reasoning processes (Nguyen et al., 2024),
which greatly undermine the reliability of LLMs in real-world applications.
To address these issues, many studies utilize knowledge graphs (KGs), which encapsulate extensive
factual information in a structured format, to improve the reasoning abilities of LLMs (Pan et al.,
2024; Luo et al., 2024). Nevertheless, because of the unstructured nature of LLMs, directly applying
them to reason on KGs is challenging.
Existing KG-enhanced LLM reasoning methods can be roughly categorized into two groups:
retrieval-based and agent-based paradigms, as shown in Figure 2 (a) and (b). Retrieval-based meth-
ods (Li et al., 2023; Yang et al., 2024b; Dehghan et al., 2024) retrieve relevant facts from KGs
with an external retriever and then feed them into the inputs of LLMs for reasoning. Agent-based
methods (Sun et al., 2024; Zhu et al., 2024; Jiang et al., 2024) treat LLMs as agents that iteratively
interact with KGs to find reasoning paths and answers.
∗Equal Contribution.
†Corresponding author.
1
arXiv:2410.13080v1  [cs.CL]  16 Oct 2024

67.0%
18.0%
15.0%
Faithful Reasoning Path
Invalid - Format Error
Invalid - Relation Error
Figure
1:
Analysis
of reasoning errors in
RoG (Luo et al., 2024).
Despite their success, retrieval-based methods require additional accurate
retrievers, which may not generalize well to unseen questions or account
for the graph structure (Mavromatis & Karypis, 2024). Conversely, agent-
based methods necessitate multiple rounds of interaction between agents
and KGs, leading to high computational costs and latency (Dehghan et al.,
2024). Furthermore, existing works still suffer from serious hallucination
issues (Agrawal et al., 2024). Sui et al. (2024) indicates that RoG (Luo
et al., 2024), a leading KG-enhanced reasoning method, still experiences
33% hallucination errors during reasoning on KGs, as shown in Figure 1.
To this end, we introduce graph-constrained reasoning (GCR), a novel KG-
guided reasoning paradigm that connects unstructured reasoning in LLMs
with structured knowledge in KGs, seeking to eliminate hallucinations dur-
ing reasoning on KGs and ensure faithful reasoning. Inspired by the con-
cept that LLMs reason through decoding (Wei et al., 2022), we incorporate
the KG structure into the LLM decoding process. This enables LLMs to directly reason on graphs
by generating reliable reasoning paths grounded in KGs that lead to correct answers.
In GCR, we first convert KG into a structured index, KG-Trie, to facilitate efficient reasoning on KG
using LLM. Trie is also known as the prefix tree (Wikipedia contributors, 2024) that compresses a
set of strings, which can be used to restrict LLM output tokens to those starting with valid prefixes
(De Cao et al., 2022; Xie et al., 2022). KG-Trie encodes the reasoning paths in KGs as formatted
strings to constrain the decoding process of LLMs. Then, we propose graph-constrained decoding
that employs a lightweight KG-specialized LLM to generate multiple KG-grounded reasoning paths
and hypothesis answers. With the constraints from KG-Trie, we ensure faithful reasoning while
leveraging the strong reasoning capabilities of LLMs to efficiently explore paths on KGs in constant
time. Finally, we input multiple generated reasoning paths and hypothesis answers into a powerful
general LLM to utilize its inductive reasoning ability to produce final answers. In this way, GCR
combines the graph reasoning strength of KG-specialized LLMs and the inductive reasoning advan-
tage in general LLMs to achieve faithful and accurate reasoning on KGs. The main contributions of
this work are as follows:
• We propose a novel framework called graph-constrained reasoning (GCR) that bridges the
gap between structured knowledge in KGs and unstructured reasoning in LLMs, allowing
for efficient reasoning on KGs via LLM decoding.
• We combine the complementary strengths of a lightweight KG-specialized LLM with a
powerful general LLM to enhance reasoning performance by leveraging their respective
graph-based reasoning and inductive reasoning capabilities.
• We conduct extensive experiments on several KGQA reasoning benchmarks, demonstrat-
ing that GCR not only achieves state-of-the-art performance with zero hallucination, but
also shows zero-shot generalizability for reasoning on unseen KGs without additional train-
ing.
2
RELATED WORK
LLM reasoning. Many studies have been proposed to analyze and improve the reasoning ability
of LLMs (Wei et al., 2022; Wang et al., 2024; Yao et al., 2024). To elicit the reasoning ability
of LLMs, Chain-of-thought (CoT) reasoning (Wei et al., 2022) prompts the model to generate a
chain of reasoning steps in response to a question. Wang et al. (2024) propose a self-consistency
mechanism that generates multiple reasoning paths and selects the most consistent answer across
them. The tree-of-thought (Yao et al., 2024) structures reasoning as a branching process, exploring
multiple steps in a tree-like structure to find optimal solutions. Other studies focus on fine-tuning
LLMs on various reasoning tasks to improve reasoning abilities (Yu et al., 2022; Hoffman et al.,
2024). For instance, OpenAI (2024c) adopts reinforcement learning to train their most advanced
LLMs called “OpenAI o1” to perform complex reasoning, which produces a long internal chain of
thought before final answers.
KG-enhanced LLM reasoning. To mitigate the knowledge gap and hallucination issues in LLM
reasoning, research incorporates KGs to enhance LLM reasoning (Pan et al., 2024). KD-CoT (Wang
2

# Reasoning Path:
# Answer:
Melania Trump
General
LLM
KG-specialized
LLM
Q
A
Question
Answer
Knowledge
Graph
(a) Retrieval-based LLM Reasoning
LLM
Reasoning
(b) Agent-based LLM Reasoning
(c) Ours: Knowledge Graph-constrained LLM Reasoning
Q: Who is
the spouse
of the ex-
president of
USA?
t=2
A: Based on the paths,
the answers are: Laura
Bush, Michelle Obama,
Melania Trump.
KG-specialized
LLM
KG-Trie
Constraint 
① Offline KG-Trie
Construction
② Graph-constrained
Decoding
t=1
Reasoning Paths and 
  Hypothesis Answers
General
LLM
③ Inductive
Reasoning
# Reasoning Path:
# Answer:
Laura Bush
# Reasoning Path:
# Answer:
Michelle Obama
Q
Knowledge
Retriever
A
Retrieved
Facts
t=1
LLM
Ex-president
Founded_in
1776
USA
Barack Obama
Born_in
Honolulu
Michelle
Obama
Sasha Obama
Morther_of
LLM Agent
A
Spouse_of
t=1
t=2
t=3
T Steps
USA
Donald Trump
Ex-president
Michelle
Obama
George W.
Bush 
Ex-president
Barack Obama
Ex-president
Spouse_of
1776
Founded_in
Washington
D.C.
Capital
Laura
Bush 
Knowledge Graph
Melania
Trump
Marry_to
Ivana
Trump
Ex-wife
Spouse_of
Q
Figure 2: Illustration of existing KG-enhanced LLM reasoning paradigms and proposed graph-
constrained reasoning (GCR). 1) First, given a KG, we convert it into the KG-Trie, serving as a
structured index to facilitate efficient reasoning path searches using LLMs. 2) Then, we design a
graph-constrained decoding process that employs a lightweight KG-specialized LLM to generate
multiple KG-grounded reasoning paths and hypothesis answers. This ensures the faithfulness of the
reasoning process while leveraging the strong capabilities of LLMs to efficiently explore reasoning
paths within KGs. 3) Finally, we input the generated reasoning paths and hypothesis answers into a
powerful general LLM to utilize its inductive reasoning ability to produce final answers.
et al., 2023) retrieve facts from an external knowledge graph to guide the CoT performed by LLMs.
RoG (Luo et al., 2024) proposes a planning-retrieval-reasoning framework that retrieves reasoning
paths from KGs to guide LLMs conducting faithful reasoning. To capture graph structure, GNN-
RAG (Mavromatis & Karypis, 2024) adopts a lightweight graph neural network to effectively re-
trieve from KGs. Instead of retrieving, StructGPT (Jiang et al., 2023) and ToG (Sun et al., 2024)
treat LLMs as agents to interact with KGs to find reasoning paths leading to the correct answers.
3
PRELIMINARY
Knowledge Graphs (KGs) represent a wealth of factual knowledge as a collection of triples: G =
{(e, r, e′) ∈E × R × E}, where E and R denote the set of entities and relations, respectively.
Reasoning Paths are sequences of consecutive triples in KGs: wz = e0
r1
−→e1
r2
−→. . .
rl
−→el,
where ∀(ei−1, ri, ei) ∈G. The paths reveal the connections between knowledge that potentially
facilitate reasoning. For example, the reasoning path: wz = Alice
marry to
−−−−−−→Bob
father of
−−−−−−→
Charlie indicates that “Alice” is married to “Bob” and “Bob” is the father of “Charlie”. Therefore,
“Alice” could be reasoned to be the mother of “Charlie”.
Knowledge Graph Question Answering (KGQA) is a representative reasoning task with the as-
sistance of KGs. Given a natural language question q and a KG G, the task aims to design a function
f to reason answers a ∈A based on knowledge from G, i.e., a = f(q, G).
3

4
APPROACH
4.1
FROM CHAIN-OF-THOUGHT REASONING TO GRAPH-CONSTRAINED REASONING
Chain-of-Thought Reasoning (CoT) (Wei et al., 2022) has been widely adopted to enhance the
reasoning ability of LLMs by autoregressively generating a series of reasoning steps leading to the
answer. Specifically, given a question q, CoT models the joint probability of the answer a and
reasoning steps z as
P(a|q) =
X
z
Pθ(a|z, q)Pθ(z|q) =
X
z
Pθ(a|q, z)
|z|
Y
i=1
Pθ(zi|q, z1:i−1),
(1)
where q denotes the input question, a denotes the final answer, θ denotes the parameters of LLMs,
and zi denotes the i-th step of the reasoning process z. To further enhance the reasoning ability,
many previous works focus on improving the reasoning process Pθ(z|q) by exploring and aggregat-
ing multiple reasoning processes (Wang et al., 2024; Yao et al., 2024).
Despite the effectiveness, a major issue remains the faithfulness of the reasoning process generated
by LLMs (Huang et al., 2024). The reasoning is represented as a sequence of tokens decoded
step-by-step, which can accumulate errors and result in hallucinated reasoning paths and answers
(Nguyen et al., 2024). To address these issues, we utilize knowledge graphs (KGs) to guide LLMs
toward faithful reasoning.
KG-enhanced Reasoning utilizes the structured knowledge in KGs to improve the reasoning of
LLMs (Luo et al., 2024; Sun et al., 2024), which can generally be expressed as finding a reasoning
path wz on KGs that connects the entities mentioned in the question and the answer. This can be
formulated as
P(a|q, G) =
X
wz
Pϕ(a|q, wz)Pϕ(wz|q, G),
(2)
where Pϕ(wz|q, G) denotes the probability of discovering a reasoning path wz on KGs G given the
question q by a function parameterized by ϕ. To acquire reasoning paths for reasoning, most prior
studies follow the retrieval-based (Li et al., 2023) or agent-based paradigm (Sun et al., 2024), as
shown in Figure 2 (a) and (b), respectively. Nevertheless, retrieval-based methods rely on precise
additional retrievers, while agent-based methods are computationally intensive and lead to high
latency. To address these issues, we propose a novel graph-constrained reasoning paradigm (GCR).
Graph-constrained Reasoning (GCR) directly incorporates KGs into the decoding process of
LLMs to achieve faithful reasoning. The overall framework of GCR is illustrated in Figure 2 (c),
which consists of three main components: 1) Knowledge Graph Trie Construction: building a
structural index of KG to guide LLM reasoning, 2) Graph-constrained Decoding: generating KG-
grounded paths and hypothesis answers using LLMs, and 3) Graph Inductive Reasoning: reasoning
over multiple paths and hypotheses to derive final answers.
4.2
KNOWLEDGE GRAPH TRIE CONSTRUCTION
Knowledge graphs (KGs) store abundant knowledge in a structured format. However, large language
models (LLMs) struggle to efficiently access and reason on KGs due to their unstructured nature. To
address this issue, we propose to convert KGs into knowledge graph Tries (KG-Tries), which serve
as a structured index of KGs to facilitate efficient reasoning on graphs using LLMs.
A Trie (a.k.a. prefix tree) (Wikipedia contributors, 2024; Fredkin, 1960) is a tree-like data structure
that stores a dynamic set of strings, where each node represents a common prefix of its children.
Tries can be used to restrict LLM output tokens to those starting with valid prefixes (De Cao et al.,
2022; Xie et al., 2022; Chen et al., 2022). The tree structure of Trie is an ideal choice for encoding
the reasoning paths in KGs for LLMs to efficiently traverse.
We first adopt the breadth-first search (BFS) algorithm to retrieve reasoning paths Wz within L hops
starting from entities mentioned in the questions. The retrieved paths are formatted as sentences
using the template shown in Figure 7. The formatted sentences are then split into tokens by the
4

tokenizer of LLM and stored as a KG-Trie CG. The overall process can be formulated as:
Wz = BFS(G, {eq}, L),
(3)
Tz = Tokenizer(Wz),
(4)
CG = Trie(Tz),
(5)
where eq denotes the entities mentioned in the question, L denotes the maximum hops of paths, and
Tz denotes the tokens of reasoning paths. The KG-Trie CG is used as a constraint to guide the LLM
decoding process.
By constructing KG-Trie for each question entity, we can enable efficient traversal of reasoning paths
in constant time (O(|Wz|)) without costly graph traversal (Sun et al., 2024). Moreover, KG-Trie can
be pre-constructed offline and loaded during reasoning. This significantly reduces the computational
cost and latency of reasoning on KGs, making it feasible for real-time applications.
============================= Prompt Input ================================
Please generate some reasoning paths in the KG starting from the topic entities to answer the question.
# Question: what is the name of justin bieber brother?
============================= LLM Output ================================
# Reasoning Path: <PATH> Justin Bieber →people.person.parents →Jeremy Bieber →peo-
ple.person.children →Jaxon Bieber </PATH>
# Answer: Jaxon Bieber
Figure 3: An example of the graph-constrained decoding. Detailed prompts can be found in Figure 8.
4.3
GRAPH-CONSTRAINED DECODING
Large language models (LLMs) have strong reasoning capabilities but still suffer from severe hal-
lucination issues, which undermines the trustworthiness of the reasoning process. To tackle this
issue, we propose graph-constrained decoding, which unifies the reasoning ability of LLMs with the
structured knowledge in KGs to generate faithful KG-grounded reasoning paths leading to answers.
Given a question q, we design an instruction prompt to harness the reasoning ability of LLMs to
generate reasoning paths wz and hypothesis answers a. To eliminate the hallucination during rea-
soning on KGs, we adopt the KG-Trie CG as constraints to guide the decoding process of LLMs and
only generate reasoning paths that are valid in KGs, formulated as:
Pϕ(a, wz|q) = Pϕ(a|q, wz)
|
{z
}
Regular decoding
Graph-constrained decoding
z
}|
{
|wz|
Y
i=1
Pϕ(wzi|q, wz1:i−1)CG(wzi|wz1:i−1),
(6)
CG(wzi|wz1:i−1) =
1, ∃prefix(wz1:i, wz), ∃wz ∈Wz,
0, else,
(7)
where wzi denotes the i-th token of the reasoning path wz, Pϕ denotes the token probabilities
predicted by the LLM with parameters ϕ, and CG(wzi|wz1:i−1) denotes the constraint function that
checks whether the generated tokens wz1:i is a valid prefix of the reasoning path using KG-Trie.
After a valid reasoning path is generated, we switch back to the regular decoding process to generate
a hypothesis answer conditioned on the path.
To further enhance KG reasoning ability, we fine-tune a lightweight KG-specialized LLM with
parameters ϕ on the graph-constrained decoding task. Specifically, given a question q, the LLM is
optimized to generate relevant reasoning paths wz that are helpful for answering the question, then
provide a hypothesis answer a based on it, which can be formulated as:
L = E(q,wz,a)∼DG log Pϕ(a, wz|q) = E

log
|a|
Y
i=1
Pϕ(ai|q, wz, a1:i−1)
|wz|
Y
j=1
Pϕ(wzj|q, wz1:j−1)

,
(8)
where ai and wzj denote the i-th token of the answer a and the j-th token of the reasoning path wz,
respectively.
5

The training data (q, wz, a) ∈DG consists of question-answer pairs and reasoning paths generated
from KGs. We use the shortest paths connecting the entities in the question and answer as the
reasoning path wz for training, where details can be found in Section 7. An example of graph-
constrained decoding is illustrated in Figure 3, where <PATH> and </PATH> are special tokens
to control the start and end of graph-constrained decoding. Experiment results in Section 5.2 show
that even a lightweight KG-specialized LLM (0.5B) can achieve satisfactory performance in KG
reasoning.
The graph-constrained decoding method differs from retrieval-based methods by integrating a pre-
constructed KG-Trie into the decoding process of LLMs. This not only reduces input tokens, but
also bridges the gap between unstructured reasoning in LLMs and structured knowledge in KGs,
allowing for efficient reasoning on KGs regardless of its scale, which results in faithful reasoning
leading to answers. Additionally, experimental results in Section 5.4 demonstrate that KG-Trie can
integrate with new KGs on the fly, showcasing its zero-shot generalizability for reasoning on unseen
KGs without further training.
4.4
GRAPH INDUCTIVE REASONING
Graph-constrained decoding harnesses the reasoning ability of a KG-specialized LLM to generate a
faithful reasoning path and a hypothesis answer. However, complex reasoning tasks typically admit
multiple reasoning paths that lead to correct answers (Stanovich et al., 2000). Incorporating diverse
reasoning paths would be beneficial for deliberate thinking and reasoning (Evans, 2010; Wang et al.,
2024). To this end, we propose to input multiple reasoning paths and hypothesis answers generated
by the KG-specialized LLM into a powerful general LLM to leverage its inductive reasoning ability
to produce final answers.
The graph-constrained decoding seamlessly integrates into the decoding process of LLMs, allowing
it to be paired with various LLM generation strategies like beam-search (Federico et al., 1995) to
take advantage of the GPU parallel computation. Thus, given a question, we adopt graph-constrained
decoding to simultaneously generate K reasoning paths and hypothesis answers with beam search in
a single LLM call, which are then inputted into a general LLM to derive final answers. The overall
process can be formulated as:
ZK = {ak, wk
z}K
k=1 = arg top-K Pϕ(a, wz|q),
(9)
Pθ(A|q, ZK) ≃
K
Y
k=1
Pθ(A|q, ak, wk
z),
(10)
where θ denotes the parameters of the general LLM, ZK denotes the set of top-K reasoning paths
and hypothesis answers, and A denotes the final answers.
We follow the FiD framework (Izacard & Grave, 2021; Singh et al., 2021) to incorporate multiple
reasoning paths and hypothesis answers to conduct inductive reasoning within one LLM call, i.e.,
Pθ(A|q, ZK), where detailed prompts can be found in Figure 9. The general LLM can be any
powerful LLM, such as ChatGPT (OpenAI, 2022), or Llama-3 (Meta, 2024), which can effectively
leverage their internal reasoning ability to reason over multiple reasoning paths to produce final
answers without additional fine-tuning.
5
EXPERIMENT
In our experiments, we aim to answer the following research questions: RQ1: Can GCR achieve
state-of-the-art reasoning performance with balances between efficiency and effectiveness? RQ2:
Can GCR eliminate hallucinations and conduct faithful reasoning? RQ3: Can GCR generalize to
unseen KGs on the fly?
5.1
EXPERIMENT SETUPS
Datasets. Following previous research (Luo et al., 2024; Sun et al., 2024), we first evaluate the
reasoning ability of GCR on two benchmark KGQA datasets: WebQuestionSP (WebQSP) (Yih et al.,
2016) and Complex WebQuestions (CWQ) (Talmor & Berant, 2018). Freebase (Bollacker et al.,
6

Table 1: Performance comparison with different baselines on the two KGQA datasets.
Types
Methods
WebQSP
CWQ
Hit
F1
Hit
F1
LLM Reasoning
Qwen2-0.5B (Yang et al., 2024a)
26.2
17.2
12.5
11.0
Qwen2-1.5B (Yang et al., 2024a)
41.3
28.0
18.5
15.7
Qwen2-7B (Yang et al., 2024a)
50.8
35.5
25.3
21.6
Llama-2-7B (Touvron et al., 2023)
56.4
36.5
28.4
21.4
Llama-3.1-8B (Meta, 2024)
55.5
34.8
28.1
22.4
GPT-4o-mini (OpenAI, 2024a)
63.8
40.5
63.8
40.5
ChatGPT (OpenAI, 2022)
59.3
43.5
34.7
30.2
ChatGPT+Few-shot (Brown et al., 2020)
68.5
38.1
38.5
28.0
ChatGPT+CoT (Wei et al., 2022)
73.5
38.5
47.5
31.0
ChatGPT+Self-Consistency (Wang et al., 2024)
83.5
63.4
56.0
48.1
Graph Reasoning
GraftNet (Sun et al., 2018)
66.7
62.4
36.8
32.7
NSM (He et al., 2021)
68.7
62.8
47.6
42.4
SR+NSM (Zhang et al., 2022)
68.9
64.1
50.2
47.1
ReaRev (Mavromatis & Karypis, 2022)
76.4
70.9
52.9
47.8
KG+LLM
KD-CoT (Wang et al., 2023)
68.6
52.5
55.7
-
EWEK-QA (Dehghan et al., 2024)
71.3
-
52.5
-
ToG (ChatGPT) (Sun et al., 2024)
76.2
-
57.6
-
ToG (GPT-4) (Sun et al., 2024)
82.6
-
68.5
-
EffiQA (Dong et al., 2024)
82.9
-
69.5
RoG (Llama-2-7B) (Luo et al., 2024)
85.7
70.8
62.6
56.2
GNN-RAG (Mavromatis & Karypis, 2024)
85.7
71.3
66.8
59.4
GNN-RAG+RA (Mavromatis & Karypis, 2024)
90.7
73.5
68.7
60.4
GCR (Llama-3.1-8B + ChatGPT)
92.6
73.2
72.7
60.9
GCR (Llama-3.1-8B + GPT-4o-mini)
92.2
74.1
75.8
61.7
2008) is adopted as the knowledge graph for both datasets. To further evaluate the generalizability
of GCR, we conduct zero-shot transfer experiments on two new KGQA datasets: CommonsenseQA
(CSQA) (Talmor et al., 2019) and MedQA-USMLE (MedQA) (Jin et al., 2021). For CSQA, we use
ConceptNet (Speer et al., 2017) as the KG, while for MedQA, we use a medical KG constructed
from the Unified Medical Language System (Yasunaga et al., 2021). The details of the datasets are
described in Section 7.
Baselines. We compare GCR with the 22 baselines grouped into three categories: 1) LLM reasoning
methods, 2) graph reasoning methods, and 3) KG-enhanced LLM reasoning methods. The detailed
baselines are listed in Section 8.
Evaluation Metrics. We adopt Hit and F1 as the evaluation metrics following previous works (Luo
et al., 2024; Sun et al., 2024) on WebQSP and CWQ. Hit checks whether any correct answer exists in
the generated predictions, while F1 considers the coverage of all answers by balancing the precision
and recall of predictions. Because CSQA and MedQA are multiple-choice QA datasets, we adopt
accuracy as the evaluation metric.
Implementations. For GCR, we use the KG-Trie to index all the reasoning paths within 2 hops
starting from question entities. For the LLMs, we use a fine-tuned Llama-3-8B (Meta, 2024) as
the KG-specialized LLM. We generate top-10 reasoning paths and hypothesis answers from graph-
constrained decoding. We adopt the advanced ChatGPT (OpenAI, 2022) and GPT-4o-mini (OpenAI,
2024a) as the general LLMs for inductive reasoning. The detailed hyperparameters and experiment
settings are described in Section 9.
5.2
RQ1: REASONING PERFORMANCE AND EFFICIENCY
Main Results. In this section, we compare GCR with other baselines on KGQA benchmarks to
evaluate the reasoning performance. From the results shown in Table 1, GCR achieves the best
performance on both datasets, outperforming the second-best by 2.1% and 9.1% in terms of Hit on
WebQSP and CWQ, respectively. The results demonstrate that GCR can effectively leverage KGs to
enhance LLMs and achieve state-of-the-art reasoning performance.
Among the LLM reasoning methods, ChatGPT with self-consistency prompts demonstrates the best
performance, which indicates the powerful reasoning ability inherent in LLMs. However, their per-
formances are still limited by the model size and complex reasoning required over structured data.
Graph reasoning methods, such as ReaRev, achieve competitive performance on WebQSP by ex-
7

Table 2: Efficiency and performance comparison of different methods on WebQSP.
Types
Methods
Hit
Avg. Runtime (s)
Avg. # LLM Calls
Avg. # LLM Tokens
Retrieval-based
S-Bert
66.9
0.87
1
293
BGE
72.7
1.05
1
357
OpenAI-Emb.
79.0
1.77
1
330
GNN-RAG
85.7
1.52
1
414
RoG
85.7
2.60
2
521
Agent-based
ToG
75.1
16.14
11.6
7,069
EffiQA
82.9
-
7.3
-
Ours
GCR
92.6
3.60
2
231
plicitly modeling the graph structure. But they struggle to generalize across different datasets and
underperform on CWQ. In KG+LLM methods, both agent-based methods (e.g., ToG, EffiQA) and
retrieval-based methods (e.g., RoG, GNN-RAG) achieve the second-best performance. Neverthe-
less, they still suffer from inefficiency and reasoning hallucinations which limit their performance.
In contrast, GCR effectively eliminates hallucinations and conducts faithful reasoning by leveraging
the structured KG index and graph-constrained decoding.
Efficiency Analysis.
To show the efficiency of GCR, we compare the average runtime, number of LLM calls, and num-
ber of input tokens with retrieval-based and agent-based methods in Table 2. For retrieval-based
methods, we compare with dense retrievers (e.g., S-Bert (Reimers & Gurevych, 2019), BGE (Zhang
et al., 2023), OpenAI-Emb. (OpenAI, 2024b)) and graph-based retrievers (e.g., GNN-RAG (Mavro-
matis & Karypis, 2024), RoG (Luo et al., 2024)), which retrieve reasoning paths from KGs and feed
them into LLMs for reasoning answers. For agent-based methods, we compare with ToG (Sun et al.,
2024) and EffiQA1 (Dong et al., 2024), which heuristically search on KGs for answers. The detailed
settings are described in Section 9.
Dense retrievers are most efficient in terms of runtime and LLM calls as they convert all paths into
sentences and encode them as embeddings in advance. However, they sacrifice their accuracy in
retrieving as they are not designed to encode graph structure. Graph-based retrievers and agent-
based methods achieve better performance by considering graph structure; however, they require
more time and LLM calls. Specifically, the retrieved graph is fed as inputs to LLMs, which leads to
a large number of input tokens. Agent-based methods, like ToG, require more LLM calls and input
tokens as the question difficulty increases due to their iterative reasoning process. In contrast, GCR
achieves the best performance with a reasonable runtime and number of LLM calls. With the help
of KG-Trie, GCR explores multiple reasoning paths at the same time during the graph-constrained
decoding, which does not involve additional LLM calls or input tokens and benefits from the parallel
GPU computation with low latency. More efficiency analysis under different beam sizes used for
graph-constrained decoding can be found in parameter analysis.
Table 3: Ablation studies of GCR on two KGQA datasets.
Variants
WebQSP
CWQ
F1
Precision
Recall
F1
Precision
Recall
GCR (Llama-3.1-8B + ChatGPT)
73.2
80.0
76.9
60.9
61.1
66.6
GCR w/o KG-specialized LLM
52.9
66.3
50.2
37.5
40.8
37.9
GCR w/o General LLM
57.0
58.0
70.1
39.4
32.8
64.3
Ablation Study. We first conduct an
ablation study to analyze the effec-
tiveness of the KG-specialized LLM
and general LLM in GCR. As shown
in Table 3, the full GCR achieves the
best performance on both datasets.
By removing the KG-specialized LLM, we feed all 2-hop reasoning paths into the general LLM.
This results in a significant performance drop, indicating its importance in utilizing reasoning abil-
ity to find relevant paths on KGs for reasoning. On the other hand, removing the general LLM
and relying solely on answers predicted by KG-specialized LLM leads to a noticeable decrease in
precision, due to noises in its predictions. This highlighting the necessity of the general LLM for
conducting inductive reasoning over multiple paths to derive final answers.
Different LLMs. We further analyze LLMs used for KG-specialized and general LLMs in Table 4.
For KG-specialized LLMs, we directly plug the KG-Trie into different LLMs to conduct graph-
constrained decoding and use the same general LLM for final reasoning. For general LLMs, we
adopt the same reasoning paths generated by KG-specialized LLMs to different LLMs to produce
final answers. For zero-shot and few-shot learning, we adopt the original LLMs without fine-tuning,
whose prompt templates can be found in Figures 8 and 10.
1Since there is no available code for EffiQA, we directly copy the results from the original paper.
8

Table 4: Comparison of different LLMs used in
GCR on WebQSP.
Components
Learning Types
Variants
Hit
F1
KG-specialized
LLM
Zero-shot
Llama-3.1-8B
28.25
10.32
Llama-3.1-70B
38.53
12.53
Few-shot
Llama-3.1-8B
33.24
11.19
Llama-3.1-70B
41.13
13.14
Fine-tuned
Qwen2-0.5B
87.48
60.03
Qwen2-1.5B
89.21
62.97
Qwen2-7B
92.31
72.74
Llama-2-7B
92.55
73.23
Llama-3.1-8B
92.74
73.14
General LLM
Zero-shot
Qwen-2-7B
86.32
67.59
Llama-3.1-8B
90.24
71.19
Llama-3.1-70B
90.24
71.19
ChatGPT
92.55
73.23
GPT-4o-mini
92.23
74.05
Results in Table 4 show that a lightweight LLM
(0.5B) can outperform a large one (70B) af-
ter fine-tuning, indicating the effectiveness of
fine-tuning in enhancing the ability of LLMs
and make them specialized for KG reason-
ing.
However, the larger LLMs (e.g., 7B
and 8B) still perform better than smaller ones,
highlighting the importance of model capac-
ity in searching relevant reasoning paths on
KGs.
Similar trends are observed in gen-
eral LLMs where larger models (e.g., GPT-4o-
mini and ChatGPT) outperform smaller ones
(e.g., Qwen-2-7B and Llama-3.1-8B), show-
casing their stronger inductive reasoning abili-
ties. This further emphasizes the need of paring
powerful general LLMs with lightweight KG-specialized LLMs to achieve better reasoning driven
by both of them.
1
3
5
10
20
Graph-constrained decoding beam size K
0
2
4
6
8
Generation Time (s)
40
50
60
70
80
90
Answer Coverage (%)
Generation Time (s)
Hit
F1
Precision
Recall
Figure 4: Parameter analysis of beam
size K.
Parameter Analysis. We first analyze the impact of dif-
ferent beam sizes K for graph-constrained decoding on
the performance of GCR. We conduct the experiments on
WebQSP with different beam sizes of 1, 3, 5, 10, and 20.
The results are shown in Figure 4. We observe that the hit
and recall of GCR increase with the beam size. Because,
with a larger beam size, the LLMs can explore more rea-
soning paths and find the correct answers. However, the
F1 score, peaks when the beam size is set to 10. This is
because the beam size of 10 can provide a balance be-
tween the exploration and exploitation of the reasoning
paths. When the beam size is set to 20, the performance
drops due to the increased complexity of the search space,
which may introduce noise and make the reasoning less
reliable. This also highlights the importance of using general LLMs to conduct inductive reason-
ing over multiple paths to disregard the noise and find the correct answers. Although the graph-
constrained decoding benefits from the parallel GPU computation to explore multiple reasoning
paths at the same time, the time cost still slightly increases from 1.4s to 7.8s with the increase of
the beam size. Thus, we set the beam size to 10 in the experiments to balance the performance and
efficiency. We also investigate the impact of L hops paths used for KG-Trie construction in Sec-
tion 10.1. The results show that GCR can achieve a good balance between reasoning performance
and efficiency by setting L = 2 and K = 10.
5.3
RQ2: HALLUCINATION ELIMINATION AND FAITHFUL REASONING
In this section, we investigate the effectiveness of KG constraints in eliminating hallucinations and
ensuring faithful reasoning. We first compare the difference of answer accuracy (Hit) and faithful
reasoning ratio by removing KG constraints in graph-constrained decoding. The faithful reasoning
ratio is calculated as the percentage of faithful reasoning in correctly predicted answers. We define
a reasoning as faithful where the generated reasoning path can be found in KGs, and vice versa.
GCR GCR w/o constraint
0
20
40
60
Answer Hit
100.0%
62.4%
WebQSP
Faithful Reasoning
Error Reasoning
GCR GCR w/o constraint
0
20
40
60
Answer Hit
100.0%
48.1%
CWQ
Figure 5: Analysis of performance and reasoning
errors in GCR.
From the Figure 5, we can observe that GCR
achieves the 100% faithful reasoning ratio on
both datasets, which indicates that GCR can
eliminate hallucinations and ensure faithful rea-
soning during reasoning on KGs. In contrast,
when removing KG constraints, both the an-
swer accuracy and faithful reasoning decrease
significantly on WebQSP. This shows that KG
constraints not only improve reasoning by re-
ducing the searching space, but also play a cru-
cial role in preventing hallucinations for accu-
9

Table 5: Examples of the faithful reasoning conducted by GCR. Red denotes the incorrect reasoning
paths and answers, while bold denotes the correct paths and answers.
Case 1: Incorrect answers and hallucinated reasoning paths without constraints.
Question
Who is niall ferguson ’s wife?
Answer
Ayaan Hirsi Ali
GCR w/o constraint
# Reasoning Path: Niall Ferguson →people.person.children →Mabel Rose Ferguson →
people.person.parents →Alyssa Mastromonaco
#Answer: Alyssa Mastromonaco
GCR
# Reasoning Path: Niall Ferguson →people.person.children →Thomas Ferguson →peo-
ple.person.parents →Ayaan Hirsi Ali
#Answer: Ayaan Hirsi Ali
Case 2: Correct answers but hallucinated reasoning paths without constraints.
Question
Where is jamarcus russell from?
Answer
Mobile
GCR w/o constraint
# Reasoning Path: JaMarcus Russell →people.person.place of birth →Tampa
#Answer: Mobile, Alabama
GCR
# Reasoning Path: JaMarcus Russell →people.person.place of birth →Mobile
#Answer: Mobile
rate reasoning. While the answer hit rate on CWQ remains almost unchanged, the ratio of faithful
reasoning still decreases to 48.1%. This implies that even if LLMs can produce correct answers, the
reasoning process is still prone to hallucinations and cannot be trusted, which is aligned with the
findings in previous studies (Nguyen et al., 2024).
Case Study. We further provide a case study to illustrate the effectiveness of GCR in eliminating
hallucinations and ensuring faithful reasoning. As shown in Table 5, the first case demonstrates that,
without constraints, the model generates an incorrect reasoning path leading to an incorrect answer
by hallucinating facts such as “Mabel Rose Ferguson is the child of Naill Ferguson and her parent
is Alyssa Mastromonaco”. In contrast, GCR generates a faithful reasoning path grounded in KGs
that “Naill Ferguson has a child named Thomas Ferguson who has a parent named Ayaan Hirsi Ali”.
Based on the paths we can reason the correct answer to the question is “Ayaan Hirsi Ali”. In the
second case, although the LLM answers the question correctly, the generated reasoning path is still
hallucinated with incorrect facts. Conversely, GCR conducts faithful reasoning with both correct
answer and reasoning path. These results demonstrate that GCR can effectively eliminate hallucina-
tions and ensure faithful reasoning by leveraging KG constraints in graph-constrained decoding.
5.4
RQ3: ZERO-SHOT GENERALIZABILITY TO UNSEEN KGS
Table 6: Zero-shot transferabil-
ity to other KGQA datasets.
Model
CSQA
MedQA
ChatGPT
79
64
GCR (ChatGPT)
85
66
GPT-4o-mini
91
75
GCR (GPT-4o-mini)
94
79
In GCR, the knowledge graph is converted into a constraint which
is plugged into the decoding process of LLMs. This allows GCR
to generalize to unseen KGs without further training. To evaluate
the generalizability of GCR, we conduct zero-shot transfer ex-
periments on two unseen KGQA datasets: CSQA (Talmor et al.,
2019) and MedQA (Jin et al., 2021). Specifically, we use the
same KG-specialized LLM (Llama-3.1-8B) trained on Freebase
as well as two general LLMs (ChatGP, GPT-4o-mini). During
reasoning, we directly plug the KG-Trie constructed from ConceptNet and medical KGs into the
GCR to conduct graph-constrained decoding without additional fine-tuning. The results are shown
in Table 6.
From the results, it is evident that GCR outperforms ChatGPT and GPT-4o-mini in zero-shot per-
formance on both datasets. Specifically, GCR shows a 7.6% increase in accuracy on CSQA and a
3.1% improvement on MedQA compared to ChatGPT. This highlights the strong zero-shot general-
izability of its graph reasoning capabilities to unseen KGs without additional training. However, the
improvement on MedQA is not as significant as that on CSQA. We hypothesize this difference may
be due to LLMs having more common sense knowledge, which aids in reasoning on common sense
knowledge graphs effectively. On the other hand, medical KGs are more specialized and require
domain-specific knowledge for reasoning, potentially limiting the generalizability of our method.
10

6
CONCLUSION
In this paper, we introduce a novel LLM reasoning paradigm called graph-constrained reasoning
(GCR) to eliminate hallucination and ensure faithful reasoning by incorporating structured KGs. To
bridge the unstructured reasoning in LLMs with the structured knowledge in KGs, we propose a
KG-Trie to encode paths in KGs using a trie-based index. KG-Trie constrains the decoding process
to guide a KG-specialized LLM to generate faithful reasoning paths grounded in KGs. By impos-
ing constraints, we can not only eliminate hallucination in reasoning but also reduce the reasoning
complexity, contributing to more efficient and accurate reasoning. Last, a powerful general LLM is
utilized as a complement to inductively reason over multiple reasoning paths to generate the final
answer. Extensive experiments demonstrate that GCR excels in faithful reasoning and generalizes
well to reason on new KGs without additional fine-tuning.
ACKNOWLEDGMENTS
We would want to express our sincere gratitude to Yuan-Fang Li for his valuable feedback and
suggestions during the preparation of this work.
REFERENCES
Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi, and Huan Liu. Mindful-rag: A study of
points of failure in retrieval augmented generation. arXiv preprint arXiv:2407.12216, 2024.
Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor. Freebase: a collab-
oratively created graph database for structuring human knowledge. In Proceedings of the 2008
ACM SIGMOD international conference on Management of data, pp. 1247–1250, 2008.
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.
Chen Chen, Yufei Wang, Bing Li, and Kwok-Yan Lam. Knowledge is flat: A seq2seq generative
framework for various knowledge graph completion. In Proceedings of the 29th International
Conference on Computational Linguistics, pp. 4005–4017, 2022.
Nicola De Cao, Gautier Izacard, Sebastian Riedel, and Fabio Petroni. Autoregressive entity retrieval.
In International Conference on Learning Representations, 2022.
Mohammad Dehghan, Mohammad Alomrani, Sunyam Bagga, David Alfonso-Hermelo, Khalil Bibi,
Abbas Ghaddar, Yingxue Zhang, Xiaoguang Li, Jianye Hao, Qun Liu, Jimmy Lin, Boxing Chen,
Prasanna Parthasarathi, Mahdi Biparva, and Mehdi Rezagholizadeh. EWEK-QA : Enhanced web
and efficient knowledge graph retrieval for citation-based question answering systems. In Lun-
Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting
of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14169–14187,
Bangkok, Thailand, August 2024. Association for Computational Linguistics. URL https:
//aclanthology.org/2024.acl-long.764.
Zixuan Dong, Baoyun Peng, Yufei Wang, Jia Fu, Xiaodong Wang, Yongxue Shan, and Xin Zhou. Ef-
fiqa: Efficient question-answering with strategic multi-model collaboration on knowledge graphs.
arXiv preprint arXiv:2406.01238, 2024.
Jonathan St BT Evans. Intuition and reasoning: A dual-process perspective. Psychological Inquiry,
21(4):313–326, 2010.
Marcello Federico, Mauro Cettolo, Fabio Brugnara, and Giuliano Antoniol. Language modelling
for efficient beam-search. Computer Speech and Language, 9(4):353–380, 1995.
Yanlin Feng, Xinyue Chen, Bill Yuchen Lin, Peifeng Wang, Jun Yan, and Xiang Ren. Scalable multi-
hop relational reasoning for knowledge-aware question answering. In Proceedings of the 2020
Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 1295–1309,
2020.
11

Edward Fredkin. Trie memory. Communications of the ACM, 3(9):490–499, 1960.
Gaole He, Yunshi Lan, Jing Jiang, Wayne Xin Zhao, and Ji-Rong Wen. Improving multi-hop knowl-
edge base question answering by learning intermediate supervision signals. In Proceedings of the
14th ACM international conference on web search and data mining, pp. 553–561, 2021.
Matthew Douglas Hoffman, Du Phan, David Dohan, Sholto Douglas, Tuan Anh Le, Aaron Parisi,
Pavel Sountsov, Charles Sutton, Sharad Vikram, and Rif A Saurous. Training chain-of-thought
via latent-variable inference. Advances in Neural Information Processing Systems, 36, 2024.
Jie Huang and Kevin Chen-Chuan Chang. Towards reasoning in large language models: A survey.
In Findings of the Association for Computational Linguistics: ACL 2023, pp. 1049–1065, 2023.
Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song,
and Denny Zhou.
Large language models cannot self-correct reasoning yet.
In The Twelfth
International Conference on Learning Representations, 2024.
Gautier Izacard and ´Edouard Grave. Leveraging passage retrieval with generative models for open
domain question answering. In Proceedings of the 16th Conference of the European Chapter of
the Association for Computational Linguistics: Main Volume, pp. 874–880, 2021.
Jinhao Jiang, Kun Zhou, Xin Zhao, and Ji-Rong Wen. Unikgqa: Unified retrieval and reasoning
for solving multi-hop question answering over knowledge graph. In The Eleventh International
Conference on Learning Representations, 2022.
Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. Structgpt: A
general framework for large language model to reason over structured data. In Proceedings of the
2023 Conference on Empirical Methods in Natural Language Processing, pp. 9237–9251, 2023.
Jinhao Jiang, Kun Zhou, Wayne Xin Zhao, Yang Song, Chen Zhu, Hengshu Zhu, and Ji-Rong
Wen. Kg-agent: An efficient autonomous agent framework for complex reasoning over knowl-
edge graph. arXiv preprint arXiv:2402.11163, 2024.
Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. What dis-
ease does this patient have? a large-scale open domain question answering dataset from medical
exams. Applied Sciences, 11(14):6421, 2021.
Shiyang Li, Yifan Gao, Haoming Jiang, Qingyu Yin, Zheng Li, Xifeng Yan, Chao Zhang, and Bing
Yin. Graph reasoning for question answering with triplet retrieval. In Findings of the Association
for Computational Linguistics: ACL 2023, pp. 3366–3375, 2023.
Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and
interpretable large language model reasoning. In International Conference on Learning Repre-
sentations, 2024.
Costas Mavromatis and George Karypis. Rearev: Adaptive reasoning for question answering over
knowledge graphs. In Findings of the Association for Computational Linguistics: EMNLP 2022,
pp. 2447–2458, 2022.
Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language model
reasoning. arXiv preprint arXiv:2405.20139, 2024.
Meta. Build the future of ai with meta llama 3, 2024. URL https://llama.meta.com/
llama3/.
Thi Nguyen, Linhao Luo, Fatemeh Shiri, Dinh Phung, Yuan-Fang Li, Thuy-Trang Vu, and Gho-
lamreza Haffari. Direct evaluation of chain-of-thought in multi-hop reasoning with knowledge
graphs. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Findings of the Association
for Computational Linguistics ACL 2024, pp. 2862–2883, Bangkok, Thailand and virtual meeting,
August 2024. Association for Computational Linguistics. URL https://aclanthology.
org/2024.findings-acl.168.
OpenAI. Introducing chatgpt, 2022. URL https://openai.com/index/chatgpt/.
12

OpenAI. Hello gpt-4o, 2024a. URL https://openai.com/index/hello-gpt-4o/.
OpenAI.
New embedding models and api updates, 2024b.
URL https://openai.com/
index/new-embedding-models-and-api-updates/.
OpenAI.
Learning to reason with llms, 2024c.
URL https://openai.com/index/
learning-to-reason-with-llms/.
Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large
language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data
Engineering (TKDE), 2024.
Shuofei Qiao, Yixin Ou, Ningyu Zhang, Xiang Chen, Yunzhi Yao, Shumin Deng, Chuanqi Tan, Fei
Huang, and Huajun Chen. Reasoning with language model prompting: A survey. In Proceedings
of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long
Papers), pp. 5368–5393, 2023.
Nils Reimers and Iryna Gurevych.
Sentence-bert: Sentence embeddings using siamese bert-
networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language
Processing. Association for Computational Linguistics, 11 2019.
URL https://arxiv.
org/abs/1908.10084.
Devendra Singh, Siva Reddy, Will Hamilton, Chris Dyer, and Dani Yogatama. End-to-end training
of multi-document reader and retriever for open-domain question answering. Advances in Neural
Information Processing Systems, 34:25968–25981, 2021.
Robyn Speer, Joshua Chin, and Catherine Havasi. Conceptnet 5.5: An open multilingual graph of
general knowledge. In Proceedings of the AAAI conference on artificial intelligence, volume 31,
2017.
KE Stanovich, RF West, and R Hertwig. Individual differences in reasoning: Implications for the ra-
tionality debate?-open peer commentary-the questionable utility of cognitive ability in explaining
cognitive illusions. 2000.
Yuan Sui, Yufei He, Nian Liu, Xiaoxin He, Kun Wang, and Bryan Hooi.
Fidelis: Faithful
reasoning in large language model for knowledge graph question answering.
arXiv preprint
arXiv:2405.13873, 2024.
Haitian Sun, Bhuwan Dhingra, Manzil Zaheer, Kathryn Mazaitis, Ruslan Salakhutdinov, and
William Cohen. Open domain question answering using early fusion of knowledge bases and
text. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Pro-
cessing, pp. 4231–4242, 2018.
Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel Ni,
Heung-Yeung Shum, and Jian Guo. Think-on-graph: Deep and responsible reasoning of large
language model on knowledge graph. In The Twelfth International Conference on Learning Rep-
resentations, 2024.
Alon Talmor and Jonathan Berant. The web as a knowledge-base for answering complex questions.
In Proceedings of the 2018 Conference of the North American Chapter of the Association for
Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp. 641–
651, 2018.
Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. Commonsenseqa: A question
answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference
of the North American Chapter of the Association for Computational Linguistics: Human Lan-
guage Technologies, Volume 1 (Long and Short Papers), pp. 4149–4158, 2019.
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Niko-
lay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open founda-
tion and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.
13

Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong,
and Zhang Xiong. Knowledge-driven cot: Exploring faithful reasoning in llms for knowledge-
intensive question answering. arXiv preprint arXiv:2308.13259, 2023.
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha
Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language
models. In The Eleventh International Conference on Learning Representations, 2024.
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in
Neural Information Processing Systems, 35:24824–24837, 2022.
Wikipedia contributors. Trie. https://en.wikipedia.org/wiki/Trie, 2024. Accessed:
2024-09-11.
Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, and S Yu Philip. A
comprehensive survey on graph neural networks. IEEE transactions on neural networks and
learning systems, 32(1):4–24, 2020.
Xin Xie, Ningyu Zhang, Zhoubo Li, Shumin Deng, Hui Chen, Feiyu Xiong, Mosha Chen, and
Huajun Chen. From discrimination to generation: Knowledge graph completion with generative
transformer. In Companion Proceedings of the Web Conference 2022, pp. 162–165, 2022.
An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li,
Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang,
Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jin Xu, Jingren Zhou, Jinze Bai,
Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng
Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai
Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan
Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Yang Fan, Yang Yao, Yichang
Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zhihao Fan. Qwen2
technical report. arXiv preprint arXiv:2407.10671, 2024a.
Rui Yang, Haoran Liu, Qingcheng Zeng, Yu He Ke, Wanxin Li, Lechao Cheng, Qingyu Chen, James
Caverlee, Yutaka Matsuo, and Irene Li. Kg-rank: Enhancing large language models for medical
qa with knowledge graphs and ranking techniques. arXiv preprint arXiv:2403.05881, 2024b.
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik
Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. Ad-
vances in Neural Information Processing Systems, 36, 2024.
Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn:
Reasoning with language models and knowledge graphs for question answering. In Proceedings
of the 2021 Conference of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies, pp. 535–546, 2021.
Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. The value
of semantic parse labeling for knowledge base question answering. In Proceedings of the 54th
Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp.
201–206, 2016.
Ping Yu, Tianlu Wang, Olga Golovneva, Badr AlKhamissi, Siddharth Verma, Zhijing Jin, Gargi
Ghosh, Mona Diab, and Asli Celikyilmaz. Alert: Adapting language models to reasoning tasks.
arXiv preprint arXiv:2212.08286, 2022.
Jing Zhang, Xiaokang Zhang, Jifan Yu, Jian Tang, Jie Tang, Cuiping Li, and Hong Chen. Subgraph
retrieval enhanced model for multi-hop knowledge base question answering. In Proceedings of the
60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers),
pp. 5773–5784, 2022.
Peitian Zhang, Shitao Xiao, Zheng Liu, Zhicheng Dou, and Jian-Yun Nie. Retrieve anything to
augment large language models. arXiv preprint arXiv:2310.07554, 2023.
14

Yuqi Zhu, Shuofei Qiao, Yixin Ou, Shumin Deng, Ningyu Zhang, Shiwei Lyu, Yue Shen, Lei Liang,
Jinjie Gu, and Huajun Chen. Knowagent: Knowledge-augmented planning for llm-based agents.
arXiv preprint arXiv:2403.03101, 2024.
Appendix
Table of Contents
7
Datasets
15
8
Baselines
16
9
Implementation Details and Experiment Settings
17
10 Additional Experiment Results
19
10.1 Performance on Different Hops . . . . . . . . . . . . . . . . . . . . . . . . . .
19
11 Templates and Prompts
19
7
DATASETS
KGQA Datasets. To compare the reasoning performance with existing methods, we use two bench-
mark KGQA datasets in this study: WebQuestionSP (WebQSP) (Yih et al., 2016) and Complex We-
bQuestions (CWQ) (Talmor & Berant, 2018). To ensure fairness, we adopt the same train and test
splits as previous works (Jiang et al., 2022; Luo et al., 2024). Details of the datasets can be found in
Table 7.
Both WebQSP and CWQ can be reasoned using Freebase KGs2 (Bollacker et al., 2008). To reduce
the size of the KGs, we use a subgraph of Freebase by extracting all triples that start from question
entities within the maximum reasoning hops provided by previous works3 (Luo et al., 2024). The
statistics of the knowledge graphs are shown in Table 9.
Fine-tuning Datasets. To enhance the KG reasoning ability of LLMs, we construct fine-tuning
datasets by generating reasoning paths from the KGs. Specifically, we adopt the training split of
WebQSP and CWQ, which contain 2,826 and 27,639 question-answer pairs, respectively. For each
question, we find all the shortest reasoning paths on KGs that connect the question entity to the
answer entity. We then convert the reasoning paths into formatted strings and pair them with the
question-answer pairs with the template shown in Figure 8 to form the fine-tuning datasets. Since
there could be multiple reasoning paths for a question, we generate multiple training instances paired
with different reasoning paths for each question-answer pair. The fine-tuning datasets contain 28,307
and 181,602 question-reasoning path-answer triples for WebQSP and CWQ, respectively. The statis-
tics of the fine-tuning datasets are shown in Table 8.
Zero-shot Generalization Datasets.
To evaluate the transferability of GCR, we further select
two new KGQA datasets4: CommonsenseQA (CSQA) (Talmor et al., 2019) and MedQA-USMLE
(MedQA) (Jin et al., 2021). CSQA is a 5-way multiple choice QA dataset that involves reasoning
with commonsense knowledge. MedQA is a 4-way multiple choice QA task that requires biomed-
ical and clinical knowledge. For CSQA, we use the ConceptNet (Speer et al., 2017), which is a
general-purpose KG that contains commonsense knowledge. For MedQA, we use a medical KG
2https://github.com/microsoft/FastRDFStore
3WebQSP: https://huggingface.co/datasets/rmanluo/RoG-webqsp, CWQ: https://
huggingface.co/datasets/rmanluo/RoG-cwq
4https://github.com/michiyasunaga/qagnn
15

constructed from the Unified Medical Language System (Yasunaga et al., 2021). The statistics of
the knowledge graphs are shown in Table 9. We respectively select 100 questions from each dataset.
For each question, following previous studies (Feng et al., 2020; Yasunaga et al., 2021), a 2-hop
subgraph is extracted from the KGs to form the zero-shot generalization datasets.
Table 7: Statistics of datasets.
Dataset
Dataset Statistics
Statistics of Answer Numbers
#Train
#Test
#Ans = 1
2 ≥#Ans ≤4
5 ≥#Ans ≤9
#Ans ≥10
WebQSP
2,826
1,628
51.2%
27.4%
8.3%
12.1%
CWQ
27,639
3,531
70.6%
19.4%
6%
4%
Table 8: Statistics of fine-tuning datasets for graph-constrained decoding.
Total
WebQSP
CWQ
209,909
28,307
181,602
Table 9: Statistics of constructed knowledge graphs.
KG
#Entities
#Relations
#Triples
Freebase
2,566,291
7,058
8,309,195
ConceptNet
799,273
17
2,151,303
MedKG
9,958
15
49,974
8
BASELINES
We compare GCR with the 22 baselines grouped into three categories: 1) LLM reasoning methods,
2) graph reasoning methods, and 3) KG-enhanced LLM reasoning methods. The details of each
baseline are described as follows.
LLM reasoning methods only rely on LLMs for reasoning without utilizing external KGs. We
include both the vanilla LLMs with different sizes and the LLMs with advanced reasoning mecha-
nisms. Specifically, we consider the following baselines:
• Qwen2-0.5B/1.5B.7B (Yang et al., 2024a) provides a series of pre-trained LLMs with dif-
ferent sizes, including 0.5B, 1.5B, and 7B parameters.
• Llama-2-7B (Touvron et al., 2023) is a large-scale LLM pre-trained on a diverse range of
tasks.
• Llama-3.1-8B (Meta, 2024) is the updated version of Llama-2 with more powerful reason-
ing capabilities.
• ChatGPT (OpenAI, 2022) is a powerful closed-source LLM that could follow instructions
to conduct complex tasks.
• GPT-4o-mini (OpenAI, 2024a) is the new flagship model of OpenAI that could reason
across different modalities and tasks.
• Few-shot prompt (Brown et al., 2020) is a few-shot learning method that provides LLMs
with a few examples in the prompts to conduct reasoning.
• CoT (Wei et al., 2022) is a chain-of-thought reasoning method that prompts LLMs to gen-
erate a chain of reasoning steps.
• Self-consistency (Wang et al., 2024) generates multiple reasoning paths and selects the
most consistent answer.
16

Graph reasoning methods focus on reasoning on KGs using graph neural networks (GNNs) (Wu
et al., 2020) or graph-based reasoning mechanisms. We include the following baselines:
• GraftNet (Sun et al., 2018) is a graph-based reasoning method that retrieves relevant sub-
graphs from KGs with entity linking.
• NSM (He et al., 2021) utilizes the sequential model to mimic the multi-hop reasoning
process on KGs.
• SR+NSM (Zhang et al., 2022) proposes a relation-path retrieval to retrieve subgraphs for
multi-hop reasoning.
• ReaRev (Mavromatis & Karypis, 2022) is a GNN-based method that reasons on KGs by
considering complex graph information.
KG-enhanced LLM reasoning methods incorporate KGs to enhance the reasoning abilities of
LLMs which can be further divided into retrieval-based and agent-based paradigms. We include the
following baselines:
Retrieval-based methods retrieve relevant facts from KGs with an external retriever and then feed
them into the inputs of LLMs for reasoning:
• KD-CoT (Wang et al., 2023) retrieves relevant knowledge from KGs to generate faithful
reasoning plans for LLMs.
• EWEK-QA (Dehghan et al., 2024) enriches the retrieved knowledge by searching from
both KGs and web.
• RoG (Luo et al., 2024) proposes a planning-retrieval-reasoning framework that retrieves
reasoning paths from KGs to guide LLMs conducting faithful reasoning.
• GNN-RAG (Mavromatis & Karypis, 2024) adopts a lightweight graph neural network to
effectively retrieve from KGs.
• GNN-RAG+RA (Mavromatis & Karypis, 2024) combines the retrieval results of both RoG
and GNN-RAG to enhance the reasoning performance.
Agent-based methods treat LLMs as agents that iteratively interact with KGs to find reasoning paths
and answers:
• ToG (Sun et al., 2024) conducts the reasoning on KGs by exploring multiple paths and
concludes the final answer by aggregating the evidence from them.
• EffiQA (Jiang et al., 2024) proposes an efficient agent-based method to reason on KGs.
9
IMPLEMENTATION DETAILS AND EXPERIMENT SETTINGS
In this section, we will detail the implementation of GCR as well as the experiment settings.
Fine-tuning KG-specialized LLMs. We fine-tune several lightweight LLMs ranging from 0.5B to
8B (Yang et al., 2024a; Touvron et al., 2023; Meta, 2024) on the fine-tuning datasets for 3 epochs.
The batch size is set to 4 and the learning rate is set to 2e-5. We use the cosine learning rate scheduler
policy with the warmup ratio set to 0.03. The training is conducted on 2 A100-80G GPUs for each
model. The training time and memory usage are shown in Table 10.
KGQA Experiment Settings. The KGQA experiment shown in Table 1 aims to compare the rea-
soning performance of GCR with existing methods. For our method, we use the fine-tuned Llama-
3.1-8B as KG-specialized LLMs, the general LLM is selected as ChatGPT and GPT-4o-mini. The
KG-Trie is constructed from the subgraph of Freebase KGs. The maximum reasoning hops are set
to 2 for both WebQSP and CWQ. The beam size is set to 10 for graph-constrained decoding. For
vanilla LLMs baselines, we use the zero-shot prompting to ask the models to answer the questions.
For other baselines, we strictly check whether the original papers follow the same settings and copy
the results for fair comparison.
Efficiency Analysis Settings. The efficiency analysis shown in Table 2 aims to compare the effi-
ciency and performance of different methods on WebQSP. For GCR, we use the same settings as the
17

Table 10: Training time and memory usage for different KG-specialized LLMs.
Model
Time
Mem. Usage per GPU
Qwen2-0.5B
3.47h
10G
Qwen2-1.5B
4.11h
25G
Qwen2-7B
14.37h
81G
Llama-2-7B
13.93h
80G
Llama-3.1-8B
14.52h
85G
KGQA experiment. For dense retriever methods (e.g., S-Bert (Reimers & Gurevych, 2019), BGE
(Zhang et al., 2023), OpenAI-Emb. (OpenAI, 2024b)), we first search all paths within 2-hops on the
KGs which are formatted as sentences with the template in Figure 7. Then, we adopt the embedding
model to encode the path sentences as embeddings which are stored in a vector database. During in-
ference, we retrieve 10 paths from the vector database with the question as query and feed them into
the LLMs for reasoning. For GNN-RAG (Mavromatis & Karypis, 2024) and RoG (Luo et al., 2024),
we strictly follow the original papers to retrieve reasoning paths and conduct the experiments. For
agent-based methods (e.g., ToG (Sun et al., 2024)), we use the same settings detailed in the original
papers. For EffiQA (Jiang et al., 2024), since there is no available code, we directly copy the results
from the original paper.
The average runtime is measured by the time taken to answer the questions. The average number
of LLM calls is the number of times the LLMs are called to answer the questions. The average
number of LLM tokens is the number of tokens inputted into LLMs to answer the questions, such
as questions and retrieved reasoning paths. The experiments are conducted on a single A100-80G
GPU for each method.
Ablation Study. In ablation study, we first try to analyze the effectiveness of different components in
GCR. We conduct the experiments on WebQSP and CWQ datasets. By removing the KG-specialized
LLM (w/o KG-specialized LLM), we search all the 2-hop paths starting from question entities and
feed them into the general LLMs for reasoning. By removing the general LLM (w/o general LLM),
we directly use the hypothesis answers generated by the KG-specialized LLMs as the final answers.
Different LLMs. We also analyze the different LLMs used for KG-specialized LLMs and general
LLMs on WebQSP. For KG-specialized LLMs, we first use the vanilla LLMs with different learning
types (i.e., zero-shot and few-shot prompting). For zero-shot prompting, we directly ask the models
to generate the reasoning paths with the constraints. For few-shot prompting, we provide the models
with a few examples in the prompts to conduct path generation. Detailed prompts can be found in
Figures 8 and 10. Then, we fine-tune the lightweight LLMs with different sizes (0.5B to 8B) on the
graph-constrained decoding task. For general LLMs, we use the vanilla LLMs to directly conduct
reasoning over multiple reasoning paths. The detailed reasoning prompts can be found in Figure 9.
Parameter Analysis. We first analyze the performance of GCR with different beam sizes for graph-
constrained decoding. We conduct the experiments on the WebQSP datasets with beam sizes of 1, 3,
5, 10, and 20. Then, we analyze the performance of GCR with different hops of paths encoded in the
KG-Trie. We conduct the experiments on the WebQSP datasets with maximum paths hops ranging
from 1 to 4.
Faithful Reasoning Analysis. We investigate the effect of the KG constraints on ensuring faithful
reasoning. We adopt the fine-tuned Llama-3.1-8B as KG-specialized LLMs. Then, we compare
the faithful reasoning rate and answer hit of GCR with and without the KG constraints in graph-
constrained decoding. The faithful reasoning rate is the percentage of the faithful reasoning in the
correctly predicted answers. A reasoning path is considered faithful if it can be found in the KGs,
and vice versa. The answer hit is the percentage of the correct answers in the predictions.
Zero-shot Generalization Analysis. We evaluate the transferability of GCR on two zero-shot gen-
eralization datasets: CSQA and MedQA. We use the fine-tuned Llama-3.1-8B as KG-specialized
LLMs and ChatGPT as well as GPT-4o-mini as the general LLMs. The KG-Trie is constructed
from the subgraph of ConceptNet and MedKG. The maximum reasoning hops are set to 2 for both
datasets. The beam size is set to 10 for graph-constrained decoding. For vanilla LLMs baselines
18

(i.e., ChatGPT and GPT-4o-mini), we use the zero-shot prompting to ask the models to answer the
questions.
10
ADDITIONAL EXPERIMENT RESULTS
10.1
PERFORMANCE ON DIFFERENT HOPS
In this section, we analyze the impact of different hops of reasoning paths on the performance of
GCR. We conduct the experiments on WebQSP with different maximum hops of reasoning paths
encoded in the KG-Trie. The results are shown in Figure 6. We observe that the performance
of GCR increases with the number of hops of reasoning paths. The performance peaks when the
maximum hops of reasoning paths are set to 2. This is because the 2-hop paths can provide sufficient
information for the LLMs to conduct reasoning. When the hops are set to 3 or 4, the performance
drops due to the increased complexity of the reasoning paths, which may introduce noise and make
the reasoning less reliable. Additionally, the size of the KG-Trie slightly increases from 0.5 MB to
7.5 MB with the increase of the hops from 1 to 4. This indicates that the KG-Trie can be efficiently
constructed with a small size and guide the LLMs to reason on graphs effectively.
1
2
3
4
KG-Trie Path Length L
0
2
4
6
Avg. KG-Trie size (MB)
60
70
80
90
Answer Coverage (%)
Avg. KG-Trie size (MB)
Hit
F1
Precision
Recall
Figure 6: Parameter analysis of path hop L for KG-Trie construction on WebQSP.
11
TEMPLATES AND PROMPTS
In this section, we illustrate all the templates and prompts used in the experiments.
Path Sentence Template. The template for converting reasoning paths into natural language sen-
tences is shown in Figure 7, where the e∗and r∗denotes the entities and relations in a reasoning
path wz = e0
r1
−→e1
r2
−→. . .
rl
−→el,
Path Sentence Template
<PATH> e1 →r1 →e2 →. . . →rl →el </PATH>
Figure 7: The template for converting reasoning paths into formatted sentences.
Graph-constrained Decoding Prompt. The prompt for graph-constrained decoding is shown in
Figure 8, where the question and mentioned entities are provided to the LLMs to generate rea-
soning paths and hypothesis answers. In the fine-tuning datasets, the supervised LLM outputs are
constructed from the ground-truth answers and reasoning paths extracted from the KGs.
19

Graph-constrained Decoding Prompt
============================= Prompt Input ================================
Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to
answer entities. Given a question, please generate some reasoning paths in the KG starting from the
topic entities to answer the question.
# Question:
<Question>
# Topic entities:
<Question Entities>
============================= LLM Output ================================
# Reasoning Path:
<PATH> <Reasoning Path> </PATH>
# Answer:
<Hypothesis Answer>
Figure 8: The prompt template for graph-constrained decoding.
The few-shot prompt template for graph-constrained decoding is shown in Figure 10. We provide a
few examples in the prompts to guide the LLMs to generate reasoning paths. Since the LLMs with
few-shot prompt learning are not fine-tuned on the graph-constrained decoding task, we only apply
the constraint to generate reasoning paths.
Graph Inductive Reasoning Prompt. The prompt for graph inductive reasoning is shown in Fig-
ure 9. We adopt the graph-constrained decoding to generate K reasoning paths and hypothesis
answers for each question. The reasoning paths and hypothesis answers are provided to the general
LLMs to answer the questions without fine-tuning.
Graph Inductive Reasoning Prompt
============================= Prompt Input ================================
# Reasoning Paths:
<Reasoning Path 1><Hypothesis Answer 1>
. . .
<Reasoning Path K><Hypothesis Answer K>
# Question:
<Question>
Based on the reasoning paths, please answer the given question. Please keep the answer as simple as
possible and only return answers. Please return each answer in a new line.
============================= LLM Output ================================
<Answer 1>
<Answer 2>
. . .
Figure 9: The prompt template for graph inductive reasoning.
20

Few-shot Graph-constrained Decoding Prompt
============================= Prompt Input ================================
Reasoning path is a sequence of triples in the KG that connects the topic entities in the question to
answer entities. Given a question, please generate some reasoning paths in the KG starting from the
topic entities to answer the question.
Example 1
# Question:
<Question>
# Topic entities:
<Question Entities>
# Reasoning Path:
<Reasoning Path>
Example 2
# Question:
<Question>
# Topic entities:
<Question Entities>
# Reasoning Path:
<Reasoning Path>
Example 3
# Question:
<Question>
# Topic entities:
<Question Entities>
# Reasoning Path:
<Reasoning Path>
Input
# Question:
<Question>
# Topic entities:
<Question Entities>
============================= LLM Output ================================
# Reasoning Path:
<Reasoning Path>
Figure 10: The few-shot prompt template for graph-constrained decoding.
21

