#include <iostream>
#include <iomanip>
#include <vector>
#include <queue>
#include <algorithm>
#include <cstdlib>
#include <time.h>

#include <cuda.h>
#include <cub/cub.cuh>

static const int BLOCK_SIZE       = 256;
static const int EDGES_PER_THREAD = 8;
static const int TILE_EDGES       = BLOCK_SIZE * EDGES_PER_THREAD;
static const int DONATION_SLOTS   = 64;
static const int INF_DIST         = 0x7FFFFFFF;

struct Edge {
    int src;
    int dst;
};

struct DonationBox {
    int begin[DONATION_SLOTS];
    int end[DONATION_SLOTS];
    int published;
    int head;
    int nslots;
};

__device__ bool in_frontier(const unsigned int *f, int v)
{
    return ((f[v >> 5] >> (v & 31)) & 1u) != 0u;
}

__device__ int expand_edge(Edge e, int *dist, unsigned int *next, int new_dist)
{
    int d = e.dst;
    if (dist[d] != INF_DIST) return 0;
    if (atomicCAS(&dist[d], INF_DIST, new_dist) != INF_DIST) return 0;
    atomicOr(&next[d >> 5], 1u << (d & 31));
    return 1;
}

__global__ void init_kernel(int *dist, unsigned int *cur, unsigned int *next,
                            int V, int words, int source)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < V) dist[i] = (i == source) ? 0 : INF_DIST;
    if (i < words) {
        cur[i]  = (i == (source >> 5)) ? (1u << (source & 31)) : 0u;
        next[i] = 0u;
    }
}

__global__ void bfs_edge_kernel(const Edge *edges, int E, int *dist,
                                const unsigned int *cur, unsigned int *next,
                                int level, int *frontier_count)
{
    typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduceT;

    __shared__ typename BlockReduceT::TempStorage reduce_tmp;
    __shared__ Edge        s_edges[TILE_EDGES];
    __shared__ DonationBox box;
    __shared__ int         s_avg;

    int tile_base = blockIdx.x * TILE_EDGES;
    int tile_size = min(TILE_EDGES, E - tile_base);

    for (int i = threadIdx.x; i < tile_size; i += BLOCK_SIZE)
        s_edges[i] = edges[tile_base + i];

    if (threadIdx.x == 0) {
        box.published = 0;
        box.head      = 0;
        box.nslots    = 0;
    }
    __syncthreads();

    int lo = threadIdx.x * EDGES_PER_THREAD;
    int hi = lo + EDGES_PER_THREAD;
    if (lo > tile_size) lo = tile_size;
    if (hi > tile_size) hi = tile_size;

    int work = 0;
    for (int i = lo; i < hi; ++i)
        work += in_frontier(cur, s_edges[i].src) ? 1 : 0;

    int total = BlockReduceT(reduce_tmp).Sum(work);
    if (threadIdx.x == 0)
        s_avg = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    __syncthreads();

    int avg   = s_avg;
    int my_hi = hi;

    if (work > avg) {
        int carried = 0;
        int mid     = lo;
        while (mid < hi && carried < avg) {
            carried += in_frontier(cur, s_edges[mid].src) ? 1 : 0;
            ++mid;
        }
        if (mid < hi) {
            int slot = atomicAdd(&box.published, 1);
            if (slot < DONATION_SLOTS) {
                box.begin[slot] = mid;
                box.end[slot]   = hi;
                my_hi           = mid;
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0)
        box.nslots = min(box.published, DONATION_SLOTS);
    __syncthreads();

    int new_dist = level + 1;
    int found    = 0;

    for (int i = lo; i < my_hi; ++i) {
        Edge e = s_edges[i];
        if (in_frontier(cur, e.src))
            found += expand_edge(e, dist, next, new_dist);
    }

    int nslots = box.nslots;
    if (nslots > 0) {
        for (;;) {
            int slot = atomicAdd(&box.head, 1);
            if (slot >= nslots) break;
            int b     = box.begin[slot];
            int e_end = box.end[slot];
            for (int i = b; i < e_end; ++i) {
                Edge e = s_edges[i];
                if (in_frontier(cur, e.src))
                    found += expand_edge(e, dist, next, new_dist);
            }
        }
    }

    __syncthreads();
    int block_found = BlockReduceT(reduce_tmp).Sum(found);
    if (threadIdx.x == 0 && block_found > 0)
        atomicAdd(frontier_count, block_found);
}

static void cpu_bfs(int V, const std::vector<int> &row, const std::vector<int> &col,
                    int source, std::vector<int> &dist)
{
    dist.assign((size_t)V, INF_DIST);
    dist[(size_t)source] = 0;
    std::queue<int> q;
    q.push(source);
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        for (int j = row[(size_t)v]; j < row[(size_t)v + 1]; ++j) {
            int u = col[(size_t)j];
            if (dist[(size_t)u] == INF_DIST) {
                dist[(size_t)u] = dist[(size_t)v] + 1;
                q.push(u);
            }
        }
    }
}

int main(int argc, char **argv)
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int V = 0, E = 0;
    if (!(std::cin >> V >> E) || V <= 0 || E <= 0) {
        std::cerr << "invalid header\n";
        return EXIT_FAILURE;
    }

    std::vector<int> esrc((size_t)E), edst((size_t)E);
    for (int i = 0; i < E; ++i) {
        int s, d;
        if (!(std::cin >> s >> d)) {
            std::cerr << "unexpected end of input at edge " << i << '\n';
            return EXIT_FAILURE;
        }
        if (s < 0 || s >= V || d < 0 || d >= V) {
            std::cerr << "edge " << i << " out of range: " << s << ' ' << d << '\n';
            return EXIT_FAILURE;
        }
        esrc[(size_t)i] = s;
        edst[(size_t)i] = d;
    }

    std::vector<int> row((size_t)V + 1, 0);
    for (int i = 0; i < E; ++i) row[(size_t)esrc[(size_t)i] + 1]++;
    for (int v = 0; v < V; ++v) row[(size_t)v + 1] += row[(size_t)v];

    std::vector<int> col((size_t)E);
    {
        std::vector<int> cursor(row.begin(), row.end() - 1);
        for (int i = 0; i < E; ++i)
            col[(size_t)cursor[(size_t)esrc[(size_t)i]]++] = edst[(size_t)i];
    }
    esrc.clear(); esrc.shrink_to_fit();
    edst.clear(); edst.shrink_to_fit();

    std::vector<Edge> edge_list((size_t)E);
    for (int v = 0; v < V; ++v)
        for (int j = row[(size_t)v]; j < row[(size_t)v + 1]; ++j) {
            edge_list[(size_t)j].src = v;
            edge_list[(size_t)j].dst = col[(size_t)j];
        }

    // int source = (argc >= 2) ? std::atoi(argv[1]) : 0;
    int source = 5;
    if (source < 0 || source >= V) {
        std::cerr << "source " << source << " out of range\n";
        return EXIT_FAILURE;
    }

    int words = (V + 31) / 32;

    Edge         *d_edges = nullptr;
    int          *d_dist  = nullptr;
    unsigned int *d_cur   = nullptr;
    unsigned int *d_next  = nullptr;
    int          *d_count = nullptr;

    cudaMalloc((void **)&d_edges, sizeof(Edge) * (size_t)E);
    cudaMalloc((void **)&d_dist, sizeof(int) * (size_t)V);
    cudaMalloc((void **)&d_cur, sizeof(unsigned int) * (size_t)words);
    cudaMalloc((void **)&d_next, sizeof(unsigned int) * (size_t)words);
    cudaMalloc((void **)&d_count, sizeof(int));

    cudaMemcpy(d_edges, edge_list.data(), sizeof(Edge) * (size_t)E,
               cudaMemcpyHostToDevice);

    int init_threads = (V > words) ? V : words;
    int init_blocks  = (init_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_kernel<<<init_blocks, BLOCK_SIZE>>>(d_dist, d_cur, d_next, V, words, source);

    int num_tiles = (E + TILE_EDGES - 1) / TILE_EDGES;

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    int level      = 0;
    int iterations = 0;
    int h_count    = 0;
    for (;;) {
        cudaMemsetAsync(d_next, 0, sizeof(unsigned int) * (size_t)words);
        cudaMemsetAsync(d_count, 0, sizeof(int));

        bfs_edge_kernel<<<num_tiles, BLOCK_SIZE>>>(d_edges, E, d_dist, d_cur,
                                                   d_next, level, d_count);

        cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
        ++iterations;
        if (h_count == 0) break;

        unsigned int *tmp = d_cur;
        d_cur  = d_next;
        d_next = tmp;
        ++level;
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    std::vector<int> gpu_dist((size_t)V);
    cudaMemcpy(gpu_dist.data(), d_dist, sizeof(int) * (size_t)V,
               cudaMemcpyDeviceToHost);

    std::vector<int> ref_dist;
    cpu_bfs(V, row, col, source, ref_dist);

    long long visited    = 0;
    int       max_level  = 0;
    int       mismatches = 0;
    for (int v = 0; v < V; ++v) {
        if (gpu_dist[(size_t)v] != ref_dist[(size_t)v]) ++mismatches;
        if (gpu_dist[(size_t)v] != INF_DIST) {
            ++visited;
            if (gpu_dist[(size_t)v] > max_level) max_level = gpu_dist[(size_t)v];
        }
    }

    std::cout << "vertices           : " << V << '\n';
    std::cout << "edges              : " << E << '\n';
    std::cout << "source             : " << source << '\n';
    std::cout << "iterations         : " << iterations << '\n';
    std::cout << "visited            : " << visited << '\n';
    std::cout << "max distance       : " << max_level << '\n';
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "kernel time (ms)   : " << ms << '\n';
    std::cout << "MTEPS              : "
              << ((ms > 0.0f) ? ((double)E * (double)iterations / ((double)ms * 1000.0)) : 0.0)
              << '\n';
    std::cout << "verification       : " << ((mismatches == 0) ? "PASS" : "FAIL") << '\n';
    if (mismatches != 0)
        std::cout << "mismatches         : " << mismatches << '\n';

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(d_edges);
    cudaFree(d_dist);
    cudaFree(d_cur);
    cudaFree(d_next);
    cudaFree(d_count);

    return (mismatches == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
