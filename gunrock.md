# Gunrock GPU Graph Analytics: Deep Research Report
## BFS, SSSP, PageRank, and Triangle Counting on the GPU

> **Source basis:** Wang et al., *Gunrock: GPU Graph Analytics*, ACM Trans. Parallel Comput. 4(2), 2017 (journal version); Wang et al., PPoPP'16 (conference version); Wang & Owens, *Fast BFS-Based Triangle Counting on GPUs*, IEEE HPEC 2019; Gunrock open-source repository (github.com/gunrock/gunrock).

---

## 1. Overview and Motivation

Gunrock is an open-source, high-performance GPU graph analytics library developed at UC Davis. Its central premise is that graph algorithms on the GPU suffer from two intertwined problems that conventional frameworks fail to address simultaneously: **irregular memory access patterns** (graphs have no spatial locality) and **irregular work distribution** (vertices in real-world graphs have wildly varying degrees, following power-law distributions).

Rather than designing an abstraction around sequential computation steps (as in Pregel or GraphLab), Gunrock centers its model on the **frontier** — a dynamically changing subset of vertices or edges that are currently active in the computation. All operations are **bulk-synchronous**: one step finishes entirely before the next begins, which is well-suited to the GPU's SIMT execution model. This avoids the fine-grained locking that would kill GPU performance while still delivering correctness.

The library represents graphs in **Compressed Sparse Row (CSR)** format by default. CSR stores neighbor lists compactly: a  array gives the start index of each vertex's neighbors in the  array. This enables coalesced memory access and allows the use of parallel *scan* and *sort* primitives to reorganize uneven workloads into uniform, GPU-friendly ones.

---

## 2. The Core Abstraction: Frontier + Four Operators

Every Gunrock algorithm is expressed as a sequence of bulk-synchronous steps over a frontier. Four operators exist:

### 2.1 Advance
Generates a **new frontier** by visiting the neighbors of the current frontier. This is the most expensive and irregular operator — different vertices have different neighbor-list sizes, producing unbalanced parallelism. Gunrock implements multiple load-balancing strategies for Advance:

- **Thread-granularity (TW):** one thread per neighbor edge — simple but degrades for high-degree vertices.
- **Warp-granularity (LB_LIGHT):** one warp per vertex — better for moderate-degree vertices.
- **Block-granularity (LB):** one CTA per high-degree vertex — handles hubs in scale-free graphs.
- **Merge-path (LB_CULL, LB_LIGHT_CULL):** maps the edge-list onto a sorted search structure so threads are assigned equal amounts of work regardless of degree — the state-of-the-art strategy based on Davidson et al.'s work.
- **ALL_EDGES:** when all edges are to be visited (e.g., PageRank), uses binary search over the full CSR row_offsets without expensive sorted-search load balancing.

The choice among these is algorithm-specific and sometimes per-iteration-dynamic.

### 2.2 Filter
Produces a **subset** of the current frontier by applying a programmer-specified predicate. Implemented via parallel compaction (prefix scan). Variants include:
- **COMPACTED_CULL:** uses several culling heuristics to aggressively prune.
- **BY_PASS:** a no-op filter for algorithms (like PageRank and CC) where no elements are removed each round.
- **LB_CULL / LB_LIGHT_CULL:** fuses the load-balanced Advance with the subsequent Filter into a single kernel, eliminating a kernel launch and a pass over the frontier — this is Gunrock's **kernel fusion** optimization.

### 2.3 Compute
A user-defined function applied in parallel to every element of the current frontier (vertices or edges). Since the parallelism here is fully regular, it is trivially mapped to GPU threads. Compute is routinely *fused* with Advance or Filter, meaning the user-defined computation runs inside the advance or filter kernel, avoiding the memory traffic of materializing an intermediate frontier.

### 2.4 Segmented Intersection
Computes the intersection of two neighbor lists for each pair of elements from two input frontiers. This is the key primitive for Triangle Counting. Gunrock implements it using a merge-path-based algorithm that is optimal for large sorted arrays on the GPU.

---

## 3. GPU-Specific Optimization Strategies

Before examining each algorithm, it is essential to understand the cross-cutting optimizations Gunrock applies.

### 3.1 Push-Pull Traversal (Direction Optimization)
Borrowed from Beamer et al.'s CPU work and adapted for the GPU. When the frontier is **small**, it is cheaper to *push* (forward traversal: for each frontier vertex, scatter to its unvisited neighbors). When the frontier is **large**, it is cheaper to *pull* (backward traversal: for each unvisited vertex, check if any of its neighbors is in the frontier). Gunrock dynamically switches between push and pull based on frontier size at each BFS/SSSP iteration. This can provide a 2x+ speedup on graphs where the frontier grows very large mid-traversal (e.g., social networks).

### 3.2 Idempotent Traversal
In BFS and SSSP, a vertex may be discovered by multiple frontier vertices simultaneously. By default, Gunrock uses  to resolve this race. The **idempotent** mode instead allows duplicate vertices in the frontier, relying on the filter step to remove them. This avoids expensive atomic operations at the cost of processing some duplicates. On graphs where the duplicate rate is low, this is a net win.

### 3.3 Two-Level Priority Queue
Used in SSSP. Rather than processing all active vertices equally, vertices are binned into a near bucket (within delta of the current minimum distance) and a far bucket. Only the near bucket is processed in the current iteration. This is the GPU realization of delta-stepping, and it dramatically reduces redundant relaxations on graphs with skewed edge-weight distributions.

### 3.4 Kernel Fusion
Gunrock's enactor specifies operations as a sequence of advance/filter/compute calls. Each such call would normally be a separate CUDA kernel launch. Gunrock exploits C++ template metaprogramming to fuse the user-defined computation functors *into* the advance or filter kernel at compile time, so that the computation happens inside the same kernel rather than in a separate pass. This eliminates the latency and memory bandwidth overhead of materializing the intermediate frontier.

### 3.5 Structure-of-Arrays (SOA) Layout
All per-vertex and per-edge data are stored in SOA format rather than Array-of-Structures (AOS). This ensures that threads in the same warp access consecutive memory locations, maximizing memory coalescing and cache utilization.

---

## 4. Breadth-First Search (BFS)

### 4.1 Problem Statement
Given an unweighted graph G and a source vertex s, assign every vertex v its shortest-hop-count distance d(v) from s.

### 4.2 Gunrock BFS Algorithm
BFS is the most fundamental Gunrock primitive. The frontier is initialized with the source vertex. Each iteration:

1. **Advance:** Expand the current vertex frontier to all unvisited neighbors. For each (src, dst, edge) in the expansion, set  and optionally . Advance uses adaptive load balancing (LB_CULL or merge-path) to handle scale-free degree distributions. Atomics () prevent multiple threads from concurrently writing to the same vertex, or optionally idempotent mode is used for higher throughput.
2. **Filter:** Remove vertices that were already visited (their label was set in a previous iteration), or in idempotent mode, deduplicate the output frontier from Advance.

The loop continues until the frontier is empty. Frontier management is done via double-buffered vertex queues stored in GPU global memory.

### 4.3 Direction Optimization in BFS
At each iteration, Gunrock measures the frontier size and the number of edges crossing into unvisited territory. If the frontier exceeds roughly half the unvisited vertices, it switches to bottom-up traversal: each unvisited vertex examines its own neighbors and marks itself if any neighbor is in the frontier. This is implemented as a separate kernel. The switch decision is made on the CPU after a lightweight frontier-size check.

### 4.4 Load Balancing Details
Gunrock adopts Merrill et al.'s adaptive strategy extended with merge-path for finer granularity:
- Small-degree vertices: one thread handles the full neighbor list.
- Medium-degree vertices: one warp (32 threads) cooperatively strips the neighbor list.
- High-degree vertices: one CTA (128–256 threads) is assigned.
- Very high-degree vertices: merge-path assigns an exactly equal number of edges per thread, independent of vertex degree.

This eliminates both under-utilization (too few threads for large hubs) and warp divergence (threads doing nothing while one thread processes a large hub).

### 4.5 Performance
Gunrock BFS achieves performance comparable to the state-of-the-art hardwired b40c implementation by Merrill et al., and provides a 1.83x throughput advantage over a 2-CPU, 2-GPU Totem configuration on social-network graphs. On scale-free graphs like , it sustains several billion traversed edges per second (GTEPS).

---

## 5. Single-Source Shortest Path (SSSP)

### 5.1 Problem Statement
Given a weighted graph G with non-negative edge weights and a source vertex s, compute the minimum-weight path d(v) from s to every other vertex v.

### 5.2 Gunrock SSSP Algorithm
SSSP is more complex than BFS because edge weights introduce the need to revisit vertices when a shorter path is found. Gunrock's SSSP is based on Davidson et al.'s delta-stepping approach, which is a parallel variant of Dijkstra's algorithm.

Each iteration maps to three Gunrock operators:

1. **Advance:** From each vertex in the current frontier, traverse all outgoing edges. For each edge (src, dst, weight), compute . If , update  using . If the update succeeds,  is added to the output frontier.
2. **Compute (fused into Advance):** Set the predecessor of  to  if the distance was updated.
3. **Filter (with Priority Queue):** Remove redundant vertices (those where a better path was found in the same iteration). Optionally apply the **two-level priority queue** to separate vertices into near (distance ≤ current_min + delta) and far buckets. This delta-stepping strategy drastically reduces the number of unnecessary edge relaxations.

The choice of delta is critical: too small means many iterations with tiny frontiers; too large means many redundant relaxations. Gunrock uses heuristics based on the graph's edge-weight distribution.

### 5.3 Workload Organization
A key insight from Davidson et al. that Gunrock incorporates: the advance step generates irregular work because neighbors have different edge weights, which puts them into different priority buckets. Gunrock's two-level priority queue explicitly reorganizes this: the frontier is partitioned into a near bin (processed immediately) and a far bin (processed in later iterations). At each super-step, only the near bin is expanded. This is the delta-stepping strategy, and Gunrock is the first framework to express it cleanly within a generic programming model via the Filter + Priority Queue combination.

### 5.4 Comparison to BFS
SSSP's edge throughput is lower than BFS because  is more expensive than , and because the priority-queue management adds overhead per iteration. Convergence also requires more iterations than BFS on the same graph. However, on road networks (high average shortest path, low average degree), SSSP benefits enormously from delta-stepping, which concentrates work on the active wavefront and avoids the full-BFS expansion that would explore every vertex even after convergence.

---

## 6. PageRank (PR)

### 6.1 Problem Statement
Assign each vertex v a score PR(v) representing its relative importance in the graph, computed iteratively as:



where d ≈ 0.85 is the damping factor. Iterations continue until the L1 norm of the change in all PR values falls below a threshold epsilon.

### 6.2 Gunrock PR Algorithm
PageRank is the most *regular* of the four algorithms because the frontier always contains **all vertices** for most of the computation. This makes it equivalent to a sparse matrix-vector multiply (SpMV), and Gunrock acknowledges that it is one of the simplest graph algorithms to implement on GPUs in terms of frontier management.

Each iteration:

1. **Initialize:** Start with a frontier containing all vertices and an initial uniform PR value (1/|V| for each vertex).
2. **Advance:** For all vertices, traverse outgoing edges. For each edge (src, dst), accumulate  into  using . The ALL_EDGES advance mode is used here — since all edges are traversed, no sorted-search load balancing is needed; binary search over the CSR row_offsets is sufficient.
3. **Compute (after Advance):** Apply the damping formula: .
4. **Filter:** Remove vertices whose PR values have already converged (change < epsilon). The BY_PASS filter mode is used initially; as more vertices converge, the active frontier shrinks and the standard compaction filter is invoked.

The loop repeats until the frontier is empty (all vertices converged).

### 6.3 GPU Implementation Notes
Because PageRank involves scattering contributions from each source vertex to all its destinations, it maps naturally to the scatter pattern. The use of  is essential for correctness since multiple source vertices may contribute to the same destination in parallel. On modern GPUs (Pascal and later), hardware atomic operations on global memory are fast enough that this is not a bottleneck.

The PR implementation is notable in Gunrock for demonstrating the framework's generality: the same advance/filter/compute operators used for traversal-based algorithms like BFS and SSSP can cleanly express a ranking algorithm whose structure is fundamentally different (no frontier growth, always-full frontier, convergence by value rather than by visitation).

### 6.4 Extensions: Personalized PageRank and Bipartite Variants
Geil et al. used Gunrock to implement Twitter's who-to-follow algorithm, incorporating Personalized PageRank (PPR), SALSA, and HITS on bipartite graphs. These variants require 2-hop traversal in a bipartite graph, which Gunrock's advance operator naturally supports by switching frontier types between vertex and edge frontiers within a single graph primitive. This demonstrated that the frontier abstraction generalizes beyond traditional graph structures.

---

## 7. Triangle Counting (TC)

### 7.1 Problem Statement
Count the number of triangles (3-cycles) in an undirected graph. This is critical for computing clustering coefficients and is a key benchmark in the Graph Challenge competition series.

### 7.2 Overview of TC Approaches
Gunrock's research group explored three methodologies:

1. **Set Intersection:** For each edge (u, v), count the number of common neighbors by intersecting the sorted adjacency lists of u and v. Sum over all edges and divide by 6 (each triangle is counted 6 times).
2. **Matrix Multiplication (SpGEMM):** Compute A^2 = A * A where A is the adjacency matrix, then count the diagonal of A^3 = A * A^2. Expensive in memory and compute.
3. **BFS-Based Subgraph Matching (Gunrock's novel method):** Traverse the graph in an all-source-BFS manner and use subgraph isomorphism to match the triangle pattern directly.

Gunrock's most competitive TC implementation uses the **set intersection** approach with Gunrock's segmented intersection operator, and the BFS-based approach for the Graph Challenge competition specifically.

### 7.3 Set Intersection TC in Gunrock
The standard Gunrock TC works as follows:

**Preprocessing:** Convert the undirected graph to a directed graph where every edge (u, v) with u < v is kept as a directed edge from lower to higher ID. This ensures each triangle is counted exactly once.

**Algorithm:**
1. **Frontier initialization:** Load the edge list as the frontier (edge-centric frontier).
2. **Segmented Intersection (Advance):** For each edge (u, v) in the frontier, compute the intersection of the neighbor list of u with the neighbor list of v. The count of common neighbors is the number of triangles containing edge (u, v). Gunrock's segmented intersection operator implements this using a merge-path-based parallel set intersection, which achieves near-optimal performance on the GPU for large sorted arrays.
3. **Compute:** Accumulate the intersection count into a global triangle counter using .
4. **No Filter needed:** All edges in the frontier are processed exactly once.

The key GPU challenge is load imbalance: edges connected to high-degree vertices have large neighbor lists to intersect, while edges between low-degree vertices have short lists. The merge-path-based intersection distributes work evenly across threads regardless of list length.

### 7.4 BFS-Based TC (Graph Challenge 2019)
Wang and Owens's 2019 work proposed an all-source-BFS method:

1. **Filtering step:** For each vertex v, compute a neighborhood encoding (NE) based on degree. Filter candidate vertices in the data graph whose NE satisfies the triangle pattern query constraints (triangle query NE ≤ data graph NE).
2. **Verification step (multi-source BFS):**
   - Initialize the frontier with all candidate vertices (all-source BFS).
   - **Advance:** BFS-traverse from each source, following edges only where the destination ID is greater than the source ID (to avoid counting each triangle 6 times). Verify edge constraints at each step.
   - **Compute (fused):** Check if newly visited vertices satisfy the stored query constraints; write valid partial matches.
   - **Filter:** Compact out invalid partial results (pruning the frontier).
3. After BFS reaches depth equal to the triangle query depth (2 hops), the partial results contain all matched triangles.

This approach achieves nearly **10 GTEPS** (giga traversed edges per second) and outperformed the 2018 Graph Challenge champion by a geometric mean of **3.84x**.

### 7.5 Performance Notes
Triangle Counting is uniquely challenging because:
- Neighbor-list intersection is inherently work-imbalanced (degree varies by orders of magnitude in real graphs).
- Memory access patterns during intersection are essentially random (intersection of two arbitrary sorted lists accessed by index).
- The output (triangle count) is a single number, so there is no output frontier to manage.

Gunrock's advantage over previous GPU TC implementations comes from the combination of efficient sorted-intersection on the GPU (merge-path) and effective work distribution across threads, combined with the ability to use aggressive filtering (NE-based pruning in the BFS approach) to eliminate unnecessary work early.

---

## 8. Cross-Algorithm GPU Optimization Summary

| Optimization | BFS | SSSP | PageRank | Triangle Counting |
|---|---|---|---|---|
| Advance load balancing | LB_CULL / merge-path | LB_CULL / merge-path | ALL_EDGES | Segmented intersection |
| Filter mode | COMPACTED_CULL | COMPACTED_CULL | BY_PASS → compact | N/A |
| Kernel fusion | Advance + Filter + Compute | Advance + Compute | Advance + Compute | Advance + Compute |
| Direction optimization | Push/Pull switch | Push only | N/A | N/A |
| Priority queue | N/A | Two-level (delta-stepping) | N/A | N/A |
| Idempotent traversal | Optional | No | N/A | N/A |
| Atomic ops | atomicCAS | atomicMin | atomicAdd | atomicAdd |

---

## 9. Performance Results (Single GPU, Tesla K40)

| Algorithm | Gunrock vs. Boost (CPU) | Gunrock vs. PowerGraph (CPU) | Gunrock vs. best hardwired GPU |
|---|---|---|---|
| BFS | ~50–100x speedup | ~10–50x speedup | ~1x (comparable) |
| SSSP | ~20–50x speedup | ~10–30x speedup | ~1x (comparable) |
| PageRank | ~30–80x speedup | ~5–15x speedup | ~1x (comparable) |
| TC | ~100x speedup | N/A | competitive |

Gunrock consistently achieves **at least an order of magnitude speedup** over Boost and PowerGraph and delivers performance comparable to the best hand-tuned GPU primitives (b40c for BFS, Davidson et al. for SSSP, Green et al. for TC).

---

## 10. The Gunrock Programming Model in Practice

A Gunrock algorithm is written in three components:

- **Problem struct:** Allocates GPU memory for algorithm-specific data (e.g., ,  for BFS;  for SSSP;  for PageRank). Provides  and  methods.
- **Functor struct:** C++ structs with static device functions , , , . These are the user-defined computations injected into the advance and filter kernels at compile time, providing kernel fusion automatically.
- **Enactor struct:** The main loop. Calls , , and optional  in sequence, specifying load-balance mode and filter mode for each call.

A minimal BFS enactor loop looks like:


The complexity of load balancing, thread assignment, memory coalescing, and frontier management is entirely hidden inside the  and  implementations.

---

## 11. Limitations and Ongoing Work

1. **Multi-GPU scaling:** The single-GPU model is well-optimized, but extending to multi-GPU requires partitioning the graph and managing communication. Gunrock has experimental multi-GPU support but it is not as mature as single-GPU.
2. **Asynchronous execution:** Gunrock is strictly BSP (bulk-synchronous). Asynchronous traversal (as in Galois) can reduce synchronization barriers and improve performance on certain graphs, but is hard to implement efficiently on GPUs due to the cost of fine-grained locking.
3. **Dynamic graphs:** The current model assumes a static graph. Dynamic graph updates (edge insertions/deletions) are not natively supported.
4. **Memory capacity:** Large graphs exceeding single-GPU VRAM require unified memory or out-of-core techniques, which are not the library's primary focus.

---

## 12. References

1. Y. Wang et al., Gunrock: GPU Graph Analytics, *ACM Trans. Parallel Comput.* 4(2), 2017.
2. Y. Wang et al., Gunrock: A High-Performance Graph Processing Library on the GPU, *PPoPP'16*, 2016.
3. L. Wang & J. D. Owens, Fast BFS-Based Triangle Counting on GPUs, *IEEE HPEC*, 2019.
4. A. Davidson et al., Work-Efficient Parallel GPU Methods for Single-Source Shortest Paths, *IPDPS'14*, 2014.
5. D. Merrill et al., Scalable GPU Graph Traversal, *PPoPP'12*, 2012.
6. S. Beamer et al., Direction-Optimizing Breadth-First Search, *SC'12*, 2012.
7. O. Green et al., GPU Triangle Counting, *HPGP'16*, 2016.
8. Gunrock open-source repository: https://github.com/gunrock/gunrock
