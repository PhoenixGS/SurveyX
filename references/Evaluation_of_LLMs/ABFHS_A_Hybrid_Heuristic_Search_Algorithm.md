# A*+BFHS: A Hybrid Heuristic Search Algorithm Zhaoxing Bu, Richard E. Korf
Computer Science Department University of California, Los Angeles Los Angeles, CA 90095 {zbu, korf}@cs.ucla.edu
Abstract
We present a new algorithm A*+BFHS for solving problems with unit-cost operators where A* and IDA* fail due to memory limitations and/or the existence of many distinct paths between the same pair of nodes. A*+BFHS is based on A* and breadth-first heuristic search (BFHS). A*+BFHS combines advantages from both algorithms, namely A*’s node ordering, BFHS’s memory savings, and both algorithms’ duplicate detection. On easy problems, A*+BFHS behaves the same as A*. On hard problems, it is slower than A* but saves a large amount of memory. Compared to BFIDA*, A*+BFHS reduces the search time and/or memory requirement by several times on a variety of planning domains.
16 Dec 
# Introduction and Overview
A* (Hart, Nilsson, and Raphael 1968) is a classic heuristic search algorithm that is used by many state-of-the-art optimal track planners (Katz et al. 2018; Franco et al. 2017, 2018; Martinez et al. 2018). One advantage of A* is duplicate detection. A* uses a Closed list and an Open list to prune duplicate nodes. A state is a unique configuration of the problem while a node is a data structure that represents a state reached by a particular path. Duplicate nodes represent the same state arrived at via different paths. The second advantage of A* is node ordering. A* always picks an Open node whose f-value is minimum among all Open nodes to expand next, which guarantees an optimal solution returned by A* when using an admissible heuristic. When using a consistent heuristic, A* expands all nodes whose f-value is less than the optimal solution cost (C∗). However, tie-breaking among nodes of equal f-value significantly affects the set of expanded nodes whose f-value equals C∗. It is common practice to choose an Open node whose h-value is minimum among all Open nodes with the same f-value, as this strategy usually leads to fewer nodes expanded. A survey of tie-breaking strategies in A* can be found in (Asai and Fukunaga 2016). A*’s main drawback is its exponential space requirement as it stores in memory all nodes generated during the search. For example, A* can fill up 8 GB of memory in a few minutes on common heuristic search and planning domains. To
solve hard problems where A* fails due to memory limitations, researchers have proposed various algorithms, usually by forgoing A*’s duplicate detection or node ordering. For example, Iterative-Deepening-A* (IDA*, Korf 1985) only has a linear memory requirement, at the price of no duplicate detection and a depth-first order within each search bound. However, IDA* may generate too many duplicate nodes on domains containing lots of distinct paths between the same pair of nodes, such as Towers of Hanoi and many planning domains, limiting its application. For example, in a grid graph, there are �m+n m � distinct shortest paths from node (0,0) to (m,n), and IDA* cannot detect these as duplicates. We introduce a new algorithm for solving problems with unit-cost operators, with many distinct paths between the same pair of nodes, where IDA* is not effective. First, we review previous algorithms. Second, we present A*+BFHS, which is based on A* and Breadth-First Heuristic Search (Zhou and Hansen 2004). Third, we present experimental results on 32 hard instances from 18 International Planning Competition (IPC) domains. On those problems, A*+BFHS is slower than A* but requires significantly less memory. Compared to BFIDA*, an algorithm that requires less memory than A*, A*+BFHS reduces the search time and/or memory requirement by several times, and sometimes by an order of magnitude, on a variety of domains.
# Previous Work
IDA* with a transposition table (IDA*+TT, Sen and Bagchi 1989; Reinefeld and Marsland 1994) uses a transposition table to detect duplicate nodes. However, IDA*+TT is outperformed by other algorithms on both heuristic search (Bu and Korf 2019) and planning domains (Zhou and Hansen 2004). A*+IDA* (Bu and Korf 2019) combines A* and IDA*, and is the state-of-the-art algorithm on the 24-Puzzle. It first runs A* until memory is almost full, then runs IDA* below each frontier node without duplicate detection. By sorting the frontier nodes with the same f-value in increasing order of h-values, A*+IDA* can significantly reduce the number of nodes generated in its last iteration. Compared to IDA*, we reported a reduction by a factor of 400 in the total number of nodes generated in the last iteration on all 50 24-Puzzle test cases in (Korf and Felner 2002). Similar to IDA*, A*+IDA* does not work well on domains with many
distinct paths between the same pair of nodes. Frontier search (Korf et al. 2005) is a family of heuristic search algorithms that work well on domains with many distinct paths between the same pair of nodes. Rather than storing all nodes generated, it stores only nodes that are at or near the search frontier, including all Open nodes and only one or two layers of Closed nodes. As a result, when a goal node is expanded, only the optimal cost is known. To reconstruct the solution path, frontier search keeps a middle layer of Closed nodes in memory. For example, we can save the Closed nodes at depth h(start)/2 as the middle layer. Each node generated below this middle layer has a pointer to its ancestor in the middle layer. After finding the optimal cost, a node in the middle layer that is on an optimal path is identified. Then the same algorithm can be applied recursively to compute the solution path from the start node to the middle node, and from the middle node to the goal node. In general, however, frontier search cannot prune all duplicates in directed graphs (Korf et al. 2005; Zhou and Hansen 2004). Divide-and-Conquer Frontier-A* (DCFA*, Korf and Zhang 2000) is a best-first frontier search based on A*. To reconstruct the solution path, DCFA* keeps a middle layer of Closed nodes that are roughly halfway along the solution path. DCFA* detects duplicates and maintains A*’s node ordering, but its memory savings compared to A* is limited on domains where the Open list is larger than the Closed list. Breadth-First Heuristic Search (BFHS, Zhou and Hansen 2004) is a frontier search algorithm for unit-cost domains. BFHS also detects duplicates but uses a breadth-first node ordering instead of A*’s best-first ordering. At first, assume the optimal cost C∗is known in advance. BFHS runs a breadth-first search (BFS) from the start node and prunes every generated node whose f-value exceeds C∗. To save memory, BFHS only keeps a few layers of nodes in memory. On undirected graphs, if we store the operators used to generate each node, and do not regenerate the parents of a node via the inverses of those operators, frontier search only needs to store two layers of nodes, the currently expanding layer and their child nodes (Korf et al. 2005). On directed graphs, one previous layer besides the above-mentioned two layers is usually stored to detect duplicates (Zhou and Hansen 2004). To reconstruct the solution path, Zhou and Hansen (2004) recommend saving the layer at the 3/4 point of the solution length as the middle layer instead of the layer at the halfway point, which usually requires more memory. As shown in (Zhou and Hansen 2004), on a domain where the Open list of A* is larger than the Closed list, BFHS usually ends up storing fewer nodes than DCFA*. In general, C∗is not known in advance. Zhou and Hansen (2004) proposed Breadth-First Iterative-Deepening-A* (BFIDA*), which runs multiple iterations of BFHS, each with a different f-bound, starting with the heuristic value of the start node. Similar to IDA*, the last iteration of BFIDA* is often significantly larger than previous iterations, so most search time is often spent on the last iteration. Compared to A*, BFHS and BFIDA* save significant memory but generate more nodes. The main drawback of BFHS and BFIDA* is that their node ordering is almost the
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/ed3a/ed3ac2b8-1a43-40a1-8e26-b92a4049ad4d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 1: An example of A*+BFHS’s search frontier. Numbers are f-values. Closed nodes are gray.</div>
worst among different node ordering schemes. BFHS and BFIDA*’s breadth-first ordering means they have to expand all nodes stored at a single depth before expanding any nodes in the next depth. As a result, they have to expand almost all nodes whose f-value equals C∗, excepting only some nodes at the same depth as the goal node, while A* may only expand a small fraction of such nodes due to its node ordering. Forward Perimeter Search (FPS, Sch¨utt, D¨obbelin, and Reinefeld 2013) builds a perimeter around the start node via BFS, then runs BFIDA* below each perimeter node. The authors only test FPS on the 24-Puzzle and 17-Pancake problem, and did not report any running time.
# A*+BFHS
# Algorithm Description
We propose A*+BFHS, a hybrid algorithm to solve problems with many paths between the same pair of nodes. A*+BFHS first runs A* until a storage threshold is reached, then runs a series of BFHS iterations on sets of frontier nodes, which are the Open nodes at the end of the A* phase. The BFHS phase can be viewed as a doubly nested loop. Each iteration of the outer loop, which we define as an iteration of the BFHS phase, corresponds to a different cost bound for BFHS. The first cost bound is set to the smallest f-value among all frontier nodes. In each iteration of the BFHS phase, we first partition the frontier nodes whose fvalue equals the cost bound into different sets according to their depths. Then the inner loop makes one call to BFHS on each set of frontier nodes, in decreasing order of their depths. This is done by initializing the BFS queue of each call to BFHS with all the nodes in the set. This inner loop continues until a solution is found or all calls to BFHS with the current bound fail to find a solution. After each call to BFHS on a set of frontier nodes, we increase the f-value of all nodes in the set to the minimum f-value of the nodes generated but not expanded in the previous call to BFHS. Figure 1 presents an example of the Open and Closed nodes at the end of the A* phase. Node S is the start node. All edge costs are 1 and the number next to each node is its f-value. Closed nodes are gray. The Open nodes B, E, F, H, I, J, K are the frontier nodes for the BFHS phase. A*+BFHS first makes a call to BFHS with a cost bound of 8 on all frontier nodes at depth 3, namely nodes H, I, J, K. If no solution is found, A*+BFHS updates the f-values of all these nodes to the minimum f-value of the nodes generated but
not expanded in that call to BFHS. A*+BFHS then makes a second call to BFHS with bound 8, starting with all frontier nodes at depth 2, namely nodes E and F. If no solution is found, A*+BFHS updates the f-values of these nodes, then makes a third call to BFHS with bound 8, starting with the frontier node B at depth 1. Suppose that no solution is found with bound 8, the updated f-values for nodes E, F, H, I, J, K are 9, and the updated f-value for node B is 10. A*+BFHS then starts a new iteration of BFHS with a cost bound of 9, making two calls to BFHS on nodes at depth 3 and 2 respectively. If the solution is found in the first call to BFHS with bound 9, BFHS will not be called again on nodes E and F. A*+BFHS is complete and admissible with admissible heuristics. A*+BFHS potentially makes calls to BFHS on all frontier nodes. When an optimal solution exists, one node on this optimal path serves as one of the start nodes for one of the calls to BFHS. Such a node is guaranteed to exist by A*’s completeness and admissibility. When the cost bound for the calls to BFHS equals C∗, the optimal solution will be found, guaranteed by BFHS’s completeness and admissibility. A state can be regenerated in separate calls to BFHS in the same iteration. To reduce such duplicates, we can decrease the number of calls to BFHS in each iteration by making each call to BFHS on a combined set of frontier nodes at adjacent depths. For the example in Figure 1, we can make one call to BFHS on the frontier nodes at depths 2 and 3 together instead of two separate calls to BFHS, by putting the frontier nodes at depth 3 after the frontier nodes at depth 2 in the initial BFS queue. In practice, we can specify a maximum number of calls to BFHS per iteration. Then in each iteration, we divide the number of depths of the frontier nodes by the number of calls to BFHS to get the number of depths for each call to BFHS. For example, if the depths of the frontier nodes range from 7 to 12 and we are limited to three calls to BFHS per iteration, each call to BFHS will start with frontier nodes at two depths. We used this strategy in our experiments. For each node generated in the BFHS phase, we check if it was generated in the A* phase. If so, we immediately prune the node if its current g-value in the BFHS phase is greater than or equal to its stored g-value in the A* phase. The primary purpose of the A* phase is to build a frontier set, so that A*+BFHS can terminate early in its last iteration. In the A* phase we have to reserve some memory for the BFHS phase. In our experiments, we first generated pattern databases or the merge-and-shrink heuristic, then allocated 1/10 of the remaining memory of 8 GB for the A* phase.
# Comparisons to BFIDA* and FPS
A*+BFHS’s BFHS phase also uses the iterative deepening concept of BFIDA*, but there are two key differences. First, in each iteration, BFIDA* always makes one call to BFHS on the start node, while we call BFHS multiple times, each on a different set of frontier nodes. Second, in each iteration, we order the frontier nodes based on their depth, and run BFHS on the deepest frontier nodes first. These differences lead to one drawback and two advantages. The drawback is that A*+BFHS may generate more
nodes than BFIDA*, as the same state can be regenerated in separate calls to BFHS in the same iteration. The first advantage is that A*+BFHS may terminate early in its last iteration. If A*+BFHS generates a goal node in the last iteration below a relatively deep frontier node, no frontier nodes above that depth will be expanded. Therefore, A*+BFHS may generate only a small number of nodes in its last iteration. In contrast, BFIDA* has to expand almost all nodes whose f-value is less than or equal to C∗in its last iteration. As a result, A*+BFHS can be faster than BFIDA*. The second advantage is that A*+BFHS’s memory usage, which is the maximum number of nodes stored during the entire search, may be smaller than that of BFIDA* for two reasons. First, the partition of frontier nodes and separate calls to BFHS within the same iteration can reduce the maximum number of nodes stored in the BFHS phase. Second, BFIDA* stores the most nodes in its last iteration while A*+BFHS may store only a small number of nodes in the last iteration due to early termination. Thus, A*+BFHS may store the most nodes in the penultimate iteration instead. FPS looks similar to A*+BFHS, but there are several fundamental differences. First, FPS builds the perimeter using a breadth-first approach while A*+BFHS builds the frontier via a best-first approach. FPS can also dynamically extend the perimeter but this approach does not always speed up the search (Sch¨utt, D¨obbelin, and Reinefeld 2013). Second, in each iteration of FPS’s BFIDA* phase, FPS makes one call to BFHS on each perimeter node. In contrast, in A*+BFHS each call to BFHS is on a set of frontier nodes. Third, FPS sorts the perimeter nodes at the same f-value using a maxtree-first or longest-path-first policy, while A*+BFHS sorts the frontier nodes at the same f-value in decreasing order of their depth. Fourth, FPS needs two separate searches for solution reconstruction while A*+BFHS only needs one.
# Solution Reconstruction
Each node generated in A*+BFHS’s BFHS phase has a pointer to its ancestral frontier node. When a goal node is generated, the solution path from the start node to the ancestral frontier node is stored in the A* phase and only one more search is needed to reconstruct the solution path from the ancestral frontier node to the goal node. This subproblem is much easier than the original problem and we can use the same heuristic function as for the original problem. Therefore, we just use A* to solve this subproblem. In addition, since we know the optimal cost of this subproblem, we can prune any node whose f-value exceeds this cost. In BFIDA*, we have to solve two subproblems to recover the paths from the start node to the middle node and from the middle node to the goal node. Zhou and Hansen (2004) called BFHS recursively to solve these two subproblems. However, pattern database heuristics (PDB, Culberson and Schaeffer 1998) only store heuristic values to the goal state, and not between arbitrary pairs of states, which complicates finding a path to a middle node. Similar to A*+BFHS, we use A* to solve the second subproblem. For the first subproblem, we use A* to compute the path from the start node to the middle node using the same
heuristic function as for the original problem, which measures the distance to the goal node, not the middle node. To save memory, we prune any node whose g-value is greater than or equal to the depth of the middle node, and any node whose f-value exceeds the optimal cost of the original problem. Since a deeper middle layer leads to more nodes stored in this approach, we saved the layer at the 1/4 point of the solution length as the middle layer instead of the 3/4 point. In this way, we do not need to build a new heuristic function for the middle node. In our experiments, the search time for solution reconstruction in BFIDA* was usually less than 1% of the total search time.
# Experimental Results and Analysis
We implemented BFIDA*, A*+IDA*, and A*+BFHS in the planner Fast Downward 20.06 (Helmert 2006), using the existing code for node expansion and heuristic functions. A*+BFHS’s A* phase reused the existing A* code. A* stores all nodes in one hash map. We used the same hash map implementation with the following difference. In each call to BFHS in both BFIDA* and A*+BFHS, we saved three layers of nodes for duplicate detection and we created one hash map for each layer of nodes. We did this because storing all nodes in one hash map in BFHS involves a lot of overhead, and is more complicated. Sch¨utt, D¨obbelin, and Reinefeld (2013) did not test FPS on planning domains and we do not know the optimal perimeter radius and sorting strategy for each domain, so we did not implement FPS. We solved about 550 problem instances from 32 unitcost domains. We present the results of A*, BFIDA*, and A*+BFHS on the 32 hardest instances. All remaining instances were easily solved by A*. We tested two A*+BFHS versions. A*+BFHS (∞) starts each call to BFHS on frontier nodes at a single depth. A*+BFHS (4) makes each call to BFHS on frontier nodes at multiple depths with at most four calls to BFHS in each iteration. All tests were run on a 3.33 GHz Intel Xeon X5680 CPU with 236 GB of RAM. We used the landmark-cut heuristic (LM-cut, Helmert and Domshlak 2009) for the satellite domain, the merge-and-shrink heuristic (M&S) with the recommended configuration (Sievers, Wehrle, and Helmert 2014, 2016; Sievers 2018) for the tpp and hiking14 domains, and the iPDB heuristic with the default configuration (Haslum et al. 2007; Sievers, Ortlieb, and Helmert 2012) for all other domains. We present the results in Table 1, where the first 26 instances are solved by A* and are sorted by A*’s running time. The last 6 instances in Table 1 are those where A* terminated without finding a solution due to the limitation of the hash map size in Fast Downward 20.06, and are sorted by BFIDA*’s running time. We ran each algorithm until it found the optimal cost and returned an optimal path. The first column gives the domain name and the instance ID. The second through fifth columns give the maximum number of nodes stored by each algorithm. For A*, this is the number of nodes stored at the end of the search. For BFIDA*, this is the largest sum of the number of nodes stored in all three layers of the search, plus the nodes stored in the 1/4 layer
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/c1ba/c1ba286d-d4e4-41df-a1d5-69c12ffac54d.png" style="width: 50%;"></div>
<div style="text-align: center;">Figure 2: A* vs. A*+BFHS in time and memory.</div>
for solution reconstruction. For A*+BFHS, this is the largest number of nodes stored in the BFHS phase plus the number of nodes stored in the A* phase. An underline means the specific algorithm needed more than 8 GB of memory to solve the problem. The last four columns are the running time in seconds, including the time for solution reconstruction but excluding the time spent on precomputing the heuristic functions, which is the same for all algorithms. For each instance, the smallest maximum number of stored nodes and shortest running time are indicated in boldface. For the last 6 instances where A* terminated without finding a solution, we report A*’s number of nodes and running time when A* terminated, with a > symbol to indicate such numbers. For A*+IDA*, we set the stored nodes threshold for the A* phase to A*+BFHS’s peak stored nodes. A*+IDA* was faster than A*+BFHS by a factor of 2 on tidybot11 18, but was slower than A*+BFHS by around 50% on mystery 14, a factor of 2 on visitall11 08-half, 4 on parking14 16 9-04, and 8 on snake18 17. Furthermore, A*+IDA* was ordersof-magnitude slower than A*+BFHS on domains such as blocks, depot, driverlog, hiking14, logistics00, pipesworldtankage, rovers, satellite, storage, termes18, and tpp. Thus we omit the results of A*+IDA* due to space limitations. We further compare the time and memory between A* and A*+BFHS in Figure 2, and between BFIDA* and A*+BFHS in Figure 3, where the x-axis is A*/BFIDA*’s peak stored nodes over A*+BFHS’s and the y-axis is A*/BFIDA*’s running time over A*+BFHS’s. Figure 2 contains the 26 instances solved by A* and Figure 3 contains all 32 instances. The red circles and green triangles correspond to A*+BFHS (4) and A*+BFHS (∞) respectively. The data points above the y = 1 line or to the right of the x = 1 line represent instances where A*+BFHS outperformed the comparison algorithm in terms of time or memory.
# A*+BFHS vs. A*
A* was the fastest on all problem instances that it solved, but also used the most memory. Among the 32 hardest problem instances we present, A* required more than 8 GB of memory on 22 instances and could not find a solution on 6 of those after running out of the hash map used by Fast
Peak Stored Nodes
Time (s)
A*+BFHS
A*+BFHS
Instance
A*
BFIDA*
(∞)
(4)
A*
BFIDA*
(∞)
(4)
depot 14
70,504,763
17,042,841
21,023,657
22,882,537
233
1,708
596
475
termes18 05
80,012,545
9,370,587
30,874,300
30,076,170
245
4,796
15,415
3,319
freecell 06
53,080,996
38,054,162
30,481,377
35,120,076
250
1,883
441
561
logistics00 14-1
57,689,357
15,441,813
19,472,255
20,169,648
255
10,381
1,160
752
driverlog 12
144,065,288
35,034,406
24,712,720
30,270,816
344
1,676
944
631
freecell 07
107,183,015
77,196,602
54,171,433
58,058,327
522
6,416
4,775
3,769
depot 11
172,447,963
27,192,174
37,977,775
46,923,423
550
3,544
7,314
4,078
tpp 11
187,011,066
93,759,836
30,856,159
33,368,912
562
7,214
9,550
2,426
mystery 14
139,924,686
135,963,227
20,302,860
20,302,860
578
7,628
839
839
tidybot11 17
69,953,936
42,080,838
33,969,968
37,090,062
662
3,684
3,223
2,694
logistics00 15-1
82,161,805
13,638,319
18,827,830
18,827,830
663
19,062
4,897
1,627
pipesworld-notankage 19
123,553,926
86,818,434
42,192,503
44,706,153
727
4,140
2,072
1,942
parking14 16 9-01
351,976,816
183,832,715
30,675,587
51,147,740
971
6,236
1,468
1,290
visitall11 08-half
407,182,291
172,474,497
34,406,966
64,671,078
1,045
4,220
2,233
1,902
tidybot11 16
115,965,857
86,095,996
41,342,908
57,026,598
1,086
5,512
2,923
3,080
snake18 08
94,699,640
44,231,998
44,081,853
51,166,308
1,131
14,877
3,445
3,192
hiking14 2-2-8
287,192,625
42,570,885
44,454,322
53,148,260
1,297
10,847
14,897
9,696
pipesworld-tankage 14
292,998,092
158,262,429
84,077,693
103,288,306
1,364
10,609
11,622
6,896
blocks 13-1
555,864,249
99,782,317
54,601,577
79,572,108
1,540
2,142
2,817
2,317
parking14 16 9-03
606,117,759
291,822,896
48,304,204
63,455,874
1,714
10,059
3,124
2,679
tidybot11 18
175,574,760
114,747,861
40,540,308
65,784,369
1,730
8,810
5,410
6,365
blocks 13-0
704,938,102
137,821,868
81,918,224
126,629,640
1,990
2,977
4,483
3,378
hiking14 2-3-6
368,433,117
124,686,777
146,623,619
148,357,537
2,480
42,379
120,494
76,603
pipesworld-notankage 20
442,232,520
301,349,348
133,708,317
148,029,967
2,693
15,245
11,499
10,629
snake18 17
265,033,991
60,041,363
56,839,243
73,365,792
3,967
20,418
8,785
8,916
satellite 08
107,395,076
20,846,202
18,870,254
19,763,323
11,834
398,884
54,551
56,296
blocks 15-0
>814,951,324
113,471,990
68,070,197
106,482,059
>2,284
3,058
4,889
3,514
storage 17
>799,907,374
397,798,456
118,138,352
133,800,503
>2,358
19,086
18,914
11,354
driverlog 15
>786,467,847
453,643,579
88,449,751
123,602,679
>1,853
24,297
15,311
8,447
rovers 09
>801,124,989
235,386,020
96,100,365
99,498,513
>2,776
25,336
42,290
16,770
rovers 11
>766,016,316
274,612,697
112,783,085
113,594,902
>2,378
26,022
43,538
16,661
parking14 16 9-04
>770,874,998
1,045,614,854
156,758,802
181,535,647
>2,306
37,701
12,304
9,813
Table 1: Instances sorted by A* running time if solved by A*. Instances where A* terminated without solving the problem (marked by >) are sorted by BFIDA* running time. An underline means more than 8 GB of memory was needed. Smalles memory and shortest times are in boldface.
Downward 20.06. On some of these instances, A* used 30 GB to 40 GB of memory before it terminated. This means A* cannot solve these 22 instances under the current IPC memory requirement, which is 8 GB. A*+BFHS required several times, sometimes an order of magnitude, less memory than A*. As a result, A*+BFHS only used more than 8 GB of memory on one instance. An interesting comparison is the space and time trade-off. For example, on parking14, A*+BFHS increased the running time by less than 100% while saving more than an order of magnitude in memory.
# A*+BFHS vs. BFIDA*
In summary, on easy problems that A*+BFHS can solve in its A* phase, A*+BFHS behaves the same as A*, and is always faster than BFIDA*. We solved around 500 such problems, which are not included here due to space limitations. On the 32 hardest problems we present, A*+BFHS is faster than BFIDA* on 27 instances and at least twice as fast on 16 of those. Furthermore, A*+BFHS requires less memory than
BFIDA* on 25 of the 32 instances and saves more than half the memory on 14 of those. In addition, these time and memory reductions exist on both the relatively easy and hard ones of the 32 instances presented, demonstrating that A*+BFHS is in general better than BFIDA* on very hard problems as well as easy problems. In the following paragraphs, we compare A*+BFHS with BFIDA* in four aspects: duplicate detection, node ordering, memory, and running time. Figure 4 compares the number of nodes generated prior to the last iteration of BFIDA* and A*+BFHS. For BFIDA*, this is the number of nodes generated in all but the last iteration. For A*+BFHS, this is the sum of nodes generated in its A* phase and all but the last iteration in its BFHS phase. The y-axis in Figure 4 is the number of nodes generated in BFIDA*’s previous iterations divided by A*+BFHS’s. We sort the 32 instances on the x-axis according to BFIDA*’s running time, so the left-most instance is the easiest and the right-most instance is the hardest for BFIDA*. The data points above the y = 1 line represent instances where
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/52b8/52b8ab73-e072-4702-8168-3a9c55df4139.png" style="width: 50%;"></div>
<div style="text-align: center;">BFIDA* peak stored # / A*+BFHS peak stored #</div>
<div style="text-align: center;">Figure 3: BFIDA* vs. A*+BFHS in time and memory.</div>
A*+BFHS generated fewer nodes than BFIDA* in the previous iterations. Compared to BFIDA*, A*+BFHS (4) generated a similar number of nodes in the previous iterations on most instances. Hiking14 2-3-6 is the only instance where A*+BFHS (4) generated at least twice as many nodes in the previous iterations as BFIDA*. However, A*+BFHS (∞) generated 2 to 7 times as many nodes in the previous iterations as BFIDA* on 11 instances. This contrast shows that, compared to BFIDA*, significantly more duplicate nodes can be generated by making each call to BFHS on frontier nodes at a single depth. However, most of those duplicate nodes can be avoided by making each call to BFHS on frontier nodes at multiple depths. A*+BFHS can generate fewer duplicate nodes than BFIDA* due to fewer BFHS iterations and making each call to BFHS on a set of frontier nodes. A*+BFHS reduced the number of nodes in previous iterations by around 50% on freecell 06 and snake18 17, and a factor of 4 on snake18 08. To our surprise, we found that on snake18 08, the number of nodes generated in the penultimate iteration of BFIDA* was twice as many as the sum of the nodes generated in A*+BFHS’s A* phase and the penultimate iteration of the BFHS phase. This means a lot of duplicate nodes were generated in BFIDA*. Snake18 generates a directed graph, in which case frontier search cannot detect all duplicate nodes (Korf et al. 2005; Zhou and Hansen 2004). Compared to BFIDA*, A*+BFHS reduced the number of nodes in the last iteration significantly, and usually by several orders of magnitude, on 28 of the 32 instances. We present this comparison in Figure 5, where the y-axis is the number of nodes generated in BFIDA*’s last iteration divided by A*+BFHS’s. We also sort the 32 instances on the x-axis according to BFIDA*’s running time, so the left-most instance is the easiest and the right-most instance is the hardest for BFIDA*. The data points above the y = 1 line represent instances where A*+BFHS generated fewer nodes than BFIDA* in the last iteration. Both A*+BFHS versions usually generated several orders of magnitude fewer nodes in the last iteration than BFIDA*, while A*+BFHS (∞) generated the fewest nodes on most instances. This large reduction proves that when ordering the frontier nodes by deepest-
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/a003/a003476f-5df8-491d-b101-fa4df90292b4.png" style="width: 50%;"></div>
<div style="text-align: center;">Test instances sorted by BFIDA* running time</div>
Figure 4: The number of nodes generated in BFIDA*’s previous iterations vs. A*+BFHS’s.
<div style="text-align: center;"><img src="https://public-pdf-extract-kit.oss-cn-shanghai.aliyuncs.com/e7ea/e7ea57dc-31fe-43a4-9a90-90501db4aa9f.png" style="width: 50%;"></div>
<div style="text-align: center;">Test instances sorted by BFIDA* running time</div>
Figure 5: The number of nodes generated in BFIDA*’s last iteration vs. A*+BFHS’s.
first, A*+BFHS can terminate early in its last iteration. On the three blocks instances and depot 11, A*+BFHS did not terminate early in its last iteration because the ancestral frontier node of the goal had a relatively low g-value. In fact, A* generated the most nodes while expanding the Open nodes whose f = C∗on the three blocks instances, which shows that node ordering is also difficult for A* on those instances. In contrast, A* generated very few nodes while expanding the Open nodes whose f = C∗on depot 11, suggesting that A*+BFHS may terminate early in its last iteration given more memory for its A* phase. A*+BFHS’s A* phase usually stored from 10 to 20 million nodes, with the exception of the snake18 domain where 40 to 50 million nodes were stored. Comparing the maximum number of stored nodes, A*+BFHS (∞) required less memory than BFIDA* on 25 instances and less than half the memory on 14 of those. For A*+BFHS (4), these two numbers are 23 and 11 respectively. In contrast, termes18 05 is the only instance where the maximum number of stored
nodes of A*+BFHS was at least twice that of BFIDA*. Comparing the two versions of A*+BFHS, A*+BFHS (4) was usually faster, sometimes significantly, due to the reduction in duplicate nodes. Compared to BFIDA*, A*+BFHS (4) was slightly slower on four instances and 80% slower on one instance. On the other 27 instances, A*+BFHS (4) was faster than BFIDA*, and at least twice as fast on 16 of those. The large speedups usually were on the instances where BFIDA* generated the most nodes in its last iteration. The best result was on the logistics00 domain, where an order of magnitude speedup was achieved. This is because BFIDA* performed very poorly on this domain due to its breadth-first node ordering. Compared to BFIDA*, A*+BFHS (∞) was slower on 11 instances and at least twice as slow on three of those, but also at least twice as fast on 12 instances. The main reason for the slower cases is the presence of many duplicate nodes generated in certain domains.
# Calling BFHS on Nodes at Multiple Depths
Comparing the two A*+BFHS versions, each has its pros and cons. A*+BFHS (4) always generated fewer duplicate nodes. Comparing the number of nodes generated in the previous iterations, A*+BFHS (∞) generated at least twice as many nodes on 7 instances. A*+BFHS (∞) generated significantly fewer nodes in the last iteration than A*+BFHS (4) on 22 instances. However, the number of nodes generated in the last iteration of A*+BFHS is usually only a small portion of the total nodes generated, so the large difference in the last iteration is not very important. A*+BFHS (4) stored a larger maximum number of nodes than A*+BFHS (∞) on almost all instances. However, the difference was usually small and never more than a factor of two. For the running time, the difference was usually less than 50%. Compared to A*+BFHS (∞), A*+BFHS (4) was faster by a factor of 3 on logistics00 15-1, 2.5 on rovers 09 and 11, 4.6 on termes18 05, 3.9 on tpp 11, and never more than 30% slower. In general, making each call to BFHS on frontier nodes at multiple depths increases both the memory usage and the number of nodes generated in the last iteration, but reduces the number of duplicate nodes and hence is often faster. Considering the memory and time trade-off, given a new problem, we recommend making each call to BFHS on frontier nodes at multiple depths. However, if we limit the number of calls to BFHS in each iteration to one, then A*+BFHS (1) will generate about the same number of nodes as BFIDA*, and early termination is no longer possible. Therefore, at least two calls should be used. So far, we have only limited BFHS to four calls in each iteration. Determining the optimal number of calls to BFHS is a subject for future work.
# Heuristic Functions and Running Time
For each node generated, A* first does duplicate checking then looks up its heuristic value if needed. Thus for each state, A* only computes its heuristic value once, no matter how many times this state is generated. However, the situation is different in BFHS. Even in a single call to BFHS, a state’s heuristic value may be calculated multiple times. For example, if a state’s f-value is greater than the cost bound of BFHS, then this state is never stored in this call to BFHS and
its heuristic value has to be computed every time it is generated. In addition, A* has only one hash map but our BFHS implementation has one hash map for each layer of nodes. Consequently, for each node generated, A* does only one hash map lookup while BFHS may have multiple lookups. Due to the above differences, the number of nodes generated per second of BFIDA* and A*+BFHS was smaller than that of A*. For the iPDB and M&S heuristics, this difference was usually less than a factor of two. For the LM-cut heuristic, A* was faster by a factor of four in terms of nodes generated per second on the satellite domain. This is because computing a node’s LM-cut heuristic is much more expensive than iPDB and M&S heuristics. This contrast shows that the choice of heuristic function also plays an important role in comparing the running time of different algorithms.
Future work includes the following. First, test A*+BFHS on more unit-cost domains. Second, investigate what is the best memory threshold for the A* phase. Third, determine the optimal number of calls to BFHS in each iteration. Fourth, find other ways to partition the frontier nodes besides the current depth-based approach. If a set of frontier nodes is too large, we may split it into multiple smaller sets and make one call to BFHS on each such smaller set. This approach may reduce the maximum number of stored nodes but may generate more duplicate nodes. In addition, when we make each call to BFHS on frontier nodes at multiple depths, we may consider the number of frontier nodes at each depth so each call to BFHS is on a different number of depths instead of a fixed number. Fifth, use external memory such as magnetic disk or flash memory in A*+BFHS to solve very hard problems. For example, instead of allocating 1/10 of RAM for the A* phase, we can first run A* until RAM is almost full, then store both Open and Closed nodes in external memory and remove them from RAM. Then in the BFHS phase, we load back the set of frontier nodes for each call to BFHS from external memory. This A*+BFHS version would never perform worse than A*, since it is identical to A* until memory is exhausted, at which point the BHFS phase would begin.
# Conclusions
We introduce a hybrid heuristic search algorithm A*+BFHS for solving problems with unit-cost operators that cannot be solved by A* due to memory limitations, nor IDA* due to the existence of many distinct paths between the same pair of nodes. A*+BFHS first runs A* until a user-specified storage threshold is reached, then runs multiple iterations of BFHS on the frontier nodes, which are the Open nodes at the end of the A* phase. Each iteration has a unique cost bound and contains multiple calls to BFHS. Each call to BFHS within the same iteration has the same cost bound but a different set of frontier nodes to start with. Within an iteration, frontier nodes are sorted deepest-first so that A*+BFHS can terminate early in its last iteration. On the around 500 easy problems solved, A*+BFHS behaves the same as A*, and is always faster than BFIDA*. On the 32 hard instances presented, A*+BFHS is slower than
A* but uses significantly less memory. A*+BFHS is faster than BFIDA* on 27 of those 32 instances and at least twice as fast on 16 of those. Furthermore, A*+BFHS requires less memory than BFIDA* on 25 of those 32 instances and saves more than half the memory on 14 of those. Another contribution of this paper is a comprehensive testing of BFIDA* on many planning domains, which is lacking in the literature.
# References
Asai, M.; and Fukunaga, A. S. 2016. Tiebreaking Strategies for A* Search: How to Explore the Final Frontier. In Proceedings of the Thirtieth AAAI Conference on Artificial Intelligence, February 12-17, 2016, Phoenix, Arizona, USA, 673–679. AAAI Press. Bu, Z.; and Korf, R. E. 2019. A*+IDA*: A Simple Hybrid Search Algorithm. In Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019, 1206–1212. ijcai.org. Culberson, J. C.; and Schaeffer, J. 1998. Pattern Databases. Comput. Intell., 14(3): 318–334. Franco, S.; Lelis, L. H.; Barley, M.; Edelkamp, S.; Martines, M.; and Moraru, I. 2018. The Complementary2 planner in the IPC 2018. IPC-9 planner abstracts, 28–31. Franco, S.; Torralba, ´A.; Lelis, L. H. S.; and Barley, M. 2017. On Creating Complementary Pattern Databases. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI 2017, Melbourne, Australia, August 19-25, 2017, 4302–4309. ijcai.org. Hart, P. E.; Nilsson, N. J.; and Raphael, B. 1968. A Formal Basis for the Heuristic Determination of Minimum Cost Paths. IEEE Trans. Syst. Sci. Cybern., 4(2): 100–107. Haslum, P.; Botea, A.; Helmert, M.; Bonet, B.; and Koenig, S. 2007. Domain-Independent Construction of Pattern Database Heuristics for Cost-Optimal Planning. In Proceedings of the Twenty-Second AAAI Conference on Artificial Intelligence, July 22-26, 2007, Vancouver, British Columbia, Canada, 1007–1012. AAAI Press. Helmert, M. 2006. The Fast Downward Planning System. J. Artif. Intell. Res., 26: 191–246. Helmert, M.; and Domshlak, C. 2009. Landmarks, Critical Paths and Abstractions: What’s the Difference Anyway? In Proceedings of the 19th International Conference on Automated Planning and Scheduling, ICAPS 2009, Thessaloniki, Greece, September 19-23, 2009. AAAI. Katz, M.; Sohrabi, S.; Samulowitz, H.; and Sievers, S. 2018. Delfi: Online planner selection for cost-optimal planning. IPC-9 planner abstracts, 57–64. Korf, R. E. 1985. Depth-First Iterative-Deepening: An Optimal Admissible Tree Search. Artif. Intell., 27(1): 97–109. Korf, R. E.; and Felner, A. 2002. Disjoint pattern database heuristics. Artif. Intell., 134(1-2): 9–22. Korf, R. E.; and Zhang, W. 2000. Divide-and-ConquerFrontier Search Applied to Optimal Sequence Alignment. In Proceedings of the Seventeenth National Conference on Artificial Intelligence and Twelfth Conference on on Innovative
Applications of Artificial Intelligence, July 30 - August 3, 2000, Austin, Texas, USA, 910–916. AAAI Press / The MIT Press. Korf, R. E.; Zhang, W.; Thayer, I.; and Hohwald, H. 2005. Frontier search. J. ACM, 52(5): 715–748. Martinez, M.; Moraru, I.; Edelkamp, S.; and Franco, S. 2018. Planning-PDBs planner in the IPC 2018. IPC-9 planner abstracts, 63–66. Reinefeld, A.; and Marsland, T. A. 1994. Enhanced Iterative-Deepening Search. IEEE Trans. Pattern Anal. Mach. Intell., 16(7): 701–710. Sch¨utt, T.; D¨obbelin, R.; and Reinefeld, A. 2013. Forward Perimeter Search with Controlled Use of Memory. In IJCAI 2013, Proceedings of the 23rd International Joint Conference on Artificial Intelligence, Beijing, China, August 3-9, 2013, 659–665. IJCAI/AAAI. Sen, A. K.; and Bagchi, A. 1989. Fast Recursive Formulations for Best-First Search That Allow Controlled Use of Memory. In Proceedings of the 11th International Joint Conference on Artificial Intelligence. Detroit, MI, USA, August 1989, 297–302. Morgan Kaufmann. Sievers, S. 2018. Merge-and-Shrink Heuristics for Classical Planning: Efficient Implementation and Partial Abstractions. In Proceedings of the Eleventh International Symposium on Combinatorial Search, SOCS 2018, Stockholm, Sweden - 14-15 July 2018, 99. AAAI Press. Sievers, S.; Ortlieb, M.; and Helmert, M. 2012. Efficient Implementation of Pattern Database Heuristics for Classical Planning. In Proceedings of the Fifth Annual Symposium on Combinatorial Search, SOCS 2012, Niagara Falls, Ontario, Canada, July 19-21, 2012. AAAI Press. Sievers, S.; Wehrle, M.; and Helmert, M. 2014. Generalized Label Reduction for Merge-and-Shrink Heuristics. In Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence, July 27 -31, 2014, Qu´ebec City, Qu´ebec, Canada, 2358–2366. AAAI Press. Sievers, S.; Wehrle, M.; and Helmert, M. 2016. An Analysis of Merge Strategies for Merge-and-Shrink Heuristics. In Proceedings of the Twenty-Sixth International Conference on Automated Planning and Scheduling, ICAPS 2016, London, UK, June 12-17, 2016, 294–298. AAAI Press. Zhou, R.; and Hansen, E. A. 2004. Breadth-First Heuristic Search. In Proceedings of the Fourteenth International Conference on Automated Planning and Scheduling (ICAPS 2004), June 3-7 2004, Whistler, British Columbia, Canada, 92–100. AAAI.
