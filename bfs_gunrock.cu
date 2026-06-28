/*
THE INPUT OF THE GRAPH WILL BE IN CSR FORMAT. THERE WILL BE A HELPER KERNEL TO CONVERT IT TP CSC AND BITMAP
FOR PULL MODE.

HAVE TO WRITE THE ENACTOR WHICH SWITCHES PUSH AND PULL.

TASK #1 -> ADVANCE PUSH MODE -> done
TASK #2 -> ADVANCE PULL MODE -> 1hr max
TASK #3 -> ENACTOR FUNCITON  -> 
TASK #4 -> FILTER -> 1-1.5 hrs max -> done
TASK #5 -> COMPUTE -> 15 mins max -> done
TASK #6(FINAL) -> GLUE THEM AND TESTING 


*/
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <bits/stdc++.h>
#include <iostream>
#include <cub/cub.cuh> // -> for prefix sum

#define EDGESPERTHREAD 32
#define THREADSPERBLOCK 256

using namespace std;
//========================================================================================================
// PUSH ADVANCE KERNEL
//========================================================================================================

__global__ void Advance_push(int *current_frontier,
                             int *outgoing_frontier,
                             int *col, int *row_indices,
                             int *prefix_sum, int edges,
                             int nothreads, int N) // N == current frontier size.
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nothreads)
    {
        int start = floor(idx * ((double)edges / nothreads));
        int end = floor((idx + 1) * ((double)edges / nothreads));
        int bstart = 0;
        int benf = N - 1;
        while (benf > bstart)
        {
            int mid = (bstart + benf + 1) / 2;
            if (prefix_sum[mid] <= start)
                bstart = mid;
            else
                benf = mid - 1;
        }

        int offset = start - prefix_sum[bstart];
        int next_ps = (bstart + 1 < N) ? prefix_sum[bstart + 1] : edges;
        int degree = next_ps - prefix_sum[bstart];
        int colIndex = row_indices[current_frontier[bstart]] + offset;
        int counter = 0;
        for (int i = start; i < end; i++)
        {
            outgoing_frontier[i] = col[colIndex];
            colIndex++;
            counter++;
            if (counter >= degree - offset)
            {
                bstart++;
                colIndex = row_indices[current_frontier[bstart]];
                int np = (bstart + 1 < N) ? prefix_sum[bstart + 1] : edges;
                degree = np - prefix_sum[bstart];
                counter = 0;
                offset = 0;
            }
        }
    }
}

//========================================================================================================
// FILTER KERNEL
//========================================================================================================
__global__ void filterBefore(int *outgoing_frontier, int total_edges, uint32_t *visited, int *keep)
{ // launch: number_edges
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_edges)
    {
        int vertex = outgoing_frontier[idx];
        int word = vertex / 32;
        int bit = vertex % 32;
        uint32_t mask = 1u << bit;
        uint32_t old = atomicOr(&visited[word], mask);
        keep[idx] = ((old & mask) == 0);
    }
}

__global__ void filterAfter(int *outgoing_frontier, int *current_frontier, int total_edges, int *prfix_keep, int *keep)
{ // launch: number_edges
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_edges)
    {
        if (keep[idx])
            current_frontier[prfix_keep[idx]] = outgoing_frontier[idx];
    }
}
//========================================================================================================
// COMPUTE KERNEL
//========================================================================================================
__global__ void compute(int *current_frontier, int *distance, int iteration, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
    {
        distance[current_frontier[idx]] = iteration;
    }
}

//========================================================================================================
// Helper Functions GPU
//========================================================================================================
void cubExclusiveScan(int *d_in, int *d_out, size_t temp_storage_bytes, void *d_temp_storage, int N)
{ // N-> size of frontier.

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        N);
        cudaDeviceSynchronize();
    
}

__device__ int degree(int *row_offsets, int vertex, int V, int E)
{
    int sub = (vertex==V-1)?E:row_offsets[vertex + 1]; 
    return sub - row_offsets[vertex];
}
//========================================================================================================
// Helper Kernels GPU
//========================================================================================================
__global__ void get_no_edges(int *current_frontier, int *row_indices, int *degree_array, int N, int V, int E)
{ //-> number of vertex in "current fronteir" (N)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        degree_array[idx] = degree(row_indices, current_frontier[idx],V,E);
}
//========================================================================================================
// Helper Functions CPU
//========================================================================================================
int getNumberEdges(int *prefix_sum_in, int *prefix_sum_out, size_t temp_storage_bytes, void *d_temp_storage, int N )
{
    if(N<=0) return 0; 
    int a1, a2;
    cudaMemcpy(&a1, &prefix_sum_in[N - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cubExclusiveScan(prefix_sum_in, prefix_sum_out, temp_storage_bytes, d_temp_storage, N);
    cudaMemcpy(&a2, &prefix_sum_out[N - 1], sizeof(int), cudaMemcpyDeviceToHost);
    return a1 + a2;
}
void input_array(int *array, int n)
{
    for (int i = 0; i < n; i++)
    {
        cin >> array[i];
    }
}

int main()
{
    int E, V, Source_node;
    cin >> E >> V >> Source_node;

    // ---------------- CPU ----------------
    int *h_col = new int[E];
    int *h_keep = new int[E];
    int *h_prekeep = new int[E];
    int *h_distance = new int[V];
    int *h_outgoing_frontier = new int[E];
    int *h_row_indices = new int[V];
    int *h_current_frontier = new int[V];
    int *h_degree_array = new int[V]; 
    int iteration = 0;
    uint32_t *visited = new uint32_t[(V + 31) / 32]();

    input_array(h_col, E);
    input_array(h_row_indices, V);

    // Initial frontier = source
    int cf_n = 1;
    h_current_frontier[0] = Source_node;
    visited[Source_node / 32] |= 1u << (Source_node % 32);

    // ---------------- GPU ----------------
    int *d_col;
    int *d_row_indices;
    int *d_current_frontier;
    int *d_degree_array;
    int *d_outgoing_frontier;
    int *d_keep;
    int *d_prekeep;
    int *d_distance;

    uint32_t *d_visited;

    cudaMalloc(&d_row_indices, (V) * sizeof(int));
    cudaMalloc(&d_distance, (V) * sizeof(int));
    cudaMalloc(&d_current_frontier, V * sizeof(int));
    cudaMalloc(&d_degree_array, V * sizeof(int));

    cudaMalloc(&d_col, E * sizeof(int));
    cudaMalloc(&d_outgoing_frontier, E * sizeof(int));
    cudaMalloc(&d_keep, (E) * sizeof(int));
    cudaMalloc(&d_prekeep, (E) * sizeof(int));
    cudaMalloc(&d_visited, ((V + 31) / 32) * sizeof(uint32_t));

    cudaMemset(d_distance, -1, V * sizeof(int));
    cudaMemset(d_keep, 0, E * sizeof(int));
    cudaMemset(d_degree_array, 0, V * sizeof(int));
    cudaMemset(d_outgoing_frontier, 0, E * sizeof(int));
    cudaMemset(d_prekeep, 0, E * sizeof(int));
    cudaMemset(d_visited, 0, ((V + 31) / 32) * sizeof(uint32_t));

    cudaMemcpy(d_col, h_col, E * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices,h_row_indices,(V) * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_current_frontier,h_current_frontier,cf_n * sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited,visited,((V + 31) / 32) * sizeof(uint32_t),cudaMemcpyHostToDevice);

    //==========================================================
    // Allocate CUB temporary storage ONCE
    //==========================================================

    size_t temp_storage_bytes = 0;

    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_storage_bytes,
        d_degree_array,
        d_degree_array,
        V); // maximum possible scan size

    void *d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    //==========================================================
    // Workflow
    //==========================================================
    clock_t starttt = clock();
    while (cf_n>0)
    {
        int blocks = (cf_n + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
        get_no_edges<<<blocks, THREADSPERBLOCK>>>(d_current_frontier, d_row_indices, d_degree_array, cf_n,V,E);
        int number_of_edges = getNumberEdges(d_degree_array, d_degree_array, temp_storage_bytes, d_temp_storage, cf_n);
        // cout<<"FIRST: "<<cf_n<<" "<<number_of_edges<<endl;    
        int totalThreads = (number_of_edges + EDGESPERTHREAD - 1) / EDGESPERTHREAD;
        int grid = (totalThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
        Advance_push<<<grid, THREADSPERBLOCK>>>(d_current_frontier, d_outgoing_frontier, d_col, d_row_indices, d_degree_array, number_of_edges, totalThreads, cf_n);
        filterBefore<<<grid, THREADSPERBLOCK>>>(d_outgoing_frontier, number_of_edges, d_visited, d_keep);
        cf_n = getNumberEdges(d_keep, d_prekeep, temp_storage_bytes, d_temp_storage, number_of_edges);
        filterAfter<<<grid, THREADSPERBLOCK>>>(d_outgoing_frontier, d_current_frontier, number_of_edges, d_prekeep, d_keep);
        blocks = (cf_n + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
        iteration++;
        // cout<<"ITERATION: "<<iteration<<" cf_n: "<<cf_n<<" Number of edges: "<<number_of_edges<<endl;
        if(blocks>0)
        compute<<<blocks, THREADSPERBLOCK>>>(d_current_frontier, d_distance, iteration, cf_n);
        cudaDeviceSynchronize();
    }

    clock_t enddd = clock();
    printf("Time Taken (GPU): %f ms\n", (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);

    //==========================================================
    // Copy back for testing
    //==========================================================

    cudaMemcpy(h_distance,
               d_distance,
               V * sizeof(int),
               cudaMemcpyDeviceToHost);

    cout << "Outgoing Frontier:\n";

    for (int i = 0; i < V; i++)
        cout << h_distance[i] << " ";

    cout << endl;

    //==========================================================
    // Cleanup
    //==========================================================

    cudaFree(d_col);
    cudaFree(d_row_indices);
    cudaFree(d_current_frontier);
    cudaFree(d_degree_array);
    cudaFree(d_outgoing_frontier);
    cudaFree(d_temp_storage);

    delete[] h_col;
    delete[] h_row_indices;
    delete[] h_current_frontier;
    delete[] h_outgoing_frontier;

    return 0;
}
/*
19 9 1
1 3 2 4 2 5 0 4 6 1 7 5 3 8 7 7 8 0 6
0 2 4 6 9 11 14 15 17
*/

/*
300 100 3
66 97 11 14 3 11 14 22 71 84 13 40 42 47 0 45 79 86 6 21 29 30 5 27 98 1 65 68 88 90 29 53 93 48 75 81 6 7 45 48 84 13 17 45 95 14 19 37 46 49 87 31 5 81 84 92 17 65 55 47 69 7 47 59 89 33 52 68 52 8 12 37 51 68 90 46 58 91 85 87 7 29 51 64 69 97 0 8 28 57 8 28 75 20 33 35 9 24 31 47 85 95 5 6 5 14 16 17 20 22 64 95 20 36 98 19 35 44 58 59 20 58 84 30 46 51 27 33 55 96 51 35 98 2 13 8 39 77 93 26 28 73 56 0 4 10 34 33 48 16 82 15 34 86 93 79 85 4 23 27 35 43 52 71 89 8 20 45 69 70 17 0 18 36 79 93 28 31 52 70 2 8 19 61 11 13 39 62 67 77 10 57 0 1 32 77 15 42 87 96 7 53 96 99 1 16 19 37 90 28 68 10 12 31 66 91 24 73 33 51 54 61 70 76 94 35 70 8 41 55 59 25 26 65 81 83 40 7 20 38 79 33 46 64 81 88 9 43 65 63 69 82 89 15 29 74 91 34 42 52 62 70 82 83 30 86 41 54 92 13 37 87 38 55 40 43 38 92 94 31 58 71 88 31 42 56 69 85 94 19 40 6 30 20 22 68 16 37 99 71
0 2 4 5 10 14 17 18 22 25 30 33 36 41 45 51 52 56 58 59 61 65 68 69 70 75 78 80 86 90 93 96 102 104 112 115 120 121 121 123 126 130 131 133 135 139 140 142 143 147 149 151 155 157 157 165 168 170 171 175 176 180 180 184 185 190 191 192 196 200 204 209 211 216 218 225 227 231 235 236 237 241 246 249 253 257 264 266 269 269 272 274 276 279 283 289 291 293 296 299 300
*/


/*
300 100 0
61 72 22 1 51 56 35 56 97 27 90 13 58 83 27 74 93 94 53 6 26 18 21 28 32 84 92 4 34 92 46 50 68 29 63 67 95 48 62 63 49 72 3 94 5 55 59 32 53 78 96 10 41 68 70 75 81 14 21 87 90 45 78 93 0 18 19 55 33 92 8 49 88 61 66 88 99 44 56 67 3 64 85 15 78 95 1 20 84 8 51 97 13 24 50 89 16 27 44 25 61 69 60 5 39 57 0 67 91 17 64 65 14 15 75 9 16 36 8 87 59 81 26 52 70 76 94 2 46 76 82 98 40 58 62 87 2 12 18 2 10 29 26 42 85 13 19 51 56 59 63 7 38 43 95 35 15 16 25 55 9 70 11 24 65 41 86 17 29 51 92 23 56 45 61 83 88 15 25 31 33 79 89 39 64 73 79 81 83 95 20 58 75 3 45 7 16 57 68 69 2 33 46 53 67 79 91 38 65 71 9 19 47 69 42 16 35 85 3 7 25 50 61 80 89 24 96 46 18 41 60 79 0 28 44 46 65 0 3 48 64 72 81 47 83 43 21 42 1 11 73 97 50 44 46 74 78 8 28 30 45 94 34 57 20 65 69 98 33 39 54 2 53 95 57 82 96 15 40 83 3 82 29 65 17 97 37 8 24 77 12 28 36 66 13 19 23 64 84 87
0 2 3 6 9 10 11 14 18 19 21 27 30 33 37 40 42 44 47 51 57 61 64 68 70 73 77 80 83 86 89 92 96 99 102 103 106 109 112 114 115 118 120 122 127 132 136 139 140 142 145 151 155 156 160 162 165 167 171 173 177 183 190 193 195 198 200 207 210 214 215 218 224 225 227 228 230 232 237 243 245 246 248 252 253 257 260 262 264 268 271 274 277 280 282 284 286 287 290 294 300
*/

/*
49 50 0
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 49
*/