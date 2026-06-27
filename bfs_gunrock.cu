/*
THE INPUT OF THE GRAPH WILL BE IN CSR FORMAT. THERE WILL BE A HELPER KERNEL TO CONVERT IT TP CSC AND BITMAP
FOR PULL MODE.

HAVE TO WRITE THE ENACTOR WHICH SWITCHES PUSH AND PULL.

TASK #1 -> ADVANCE PUSH MODE (ONCE THIS IS DONE, REST WILL FOLLOW EASILY!!(MOTIVATION + EVERYTHING IS SETTLED IN THE BRAIN)) -> 45 mins
TASK #2 -> ADVANCE PULL MODE -> 1hrs max
TASK #3 -> ENACTOR FUNCITON
TASK #4 -> FILTER -> 1-1.5 hrs max
TASK #5 -> COMPUTE -> 15 mins max
TASK #6(FINAL) -> GLUE THEM AND TESTING

========================================================================================================
advance:
    to Have:
     - current frontier
     - length of it (N)

    get_no_edges -> degree ka array;
    cubExclusiveScan -> prefix sum wala array;
    number_of_edges = memcpy(degree_array[N-1]) + memcpy(prefixsum[N-1]);
    advance_push<<<ceil(number_of_edges/(EDGESPERTHREAD*tpb)), tpb>>> ; -> outgoing_frontier;
========================================================================================================

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

__global__ void Advance_push(int *current_frontier,
                             int *outgoing_frontier,
                             int *col, int *row_indices,
                             int *prefix_sum, int edges,
                             int nothreads, int N, int total_edges) // N == current frontier size.
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
        int next_ps = (bstart + 1 < N) ? prefix_sum[bstart + 1] : total_edges;
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
                int np = (bstart + 1 < N) ? prefix_sum[bstart + 1] : total_edges;
                degree = np - prefix_sum[bstart];
                counter = 0;
                offset = 0;
            }
        }
    }
}

//========================================================================================================
// Helper Functions GPU
//========================================================================================================
void cubExclusiveScan(int *d_in, size_t temp_storage_bytes, void *d_temp_storage, int N)
{ // N-> size of frontier.

    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_in,
        N);

    cudaDeviceSynchronize();
}

__device__ int degree(int *row_offsets, int vertex)
{
    return row_offsets[vertex + 1] - row_offsets[vertex];
}
//========================================================================================================
// Helper Kernels GPU
//========================================================================================================
__global__ void get_no_edges(int *current_frontier, int *row_indices, int *degree_array, int N)
{ //-> number of vertex in "current fronteir" (N)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        degree_array[idx] = degree(row_indices, current_frontier[idx]);
}
//========================================================================================================
// Helper Functions CPU
//========================================================================================================
int getNumberEdges(int *prefix_sum, size_t temp_storage_bytes, void *d_temp_storage, int N)
{
    int a1, a2;
    cudaMemcpy(&a1, &prefix_sum[N - 1], sizeof(int), cudaMemcpyDeviceToHost);
    cubExclusiveScan(prefix_sum, temp_storage_bytes, d_temp_storage, N);
    cudaMemcpy(&a2, &prefix_sum[N - 1], sizeof(int), cudaMemcpyDeviceToHost);
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
    int *h_row_indices = new int[V];

    input_array(h_col, E);
    input_array(h_row_indices, V);

    // Initial frontier = source
    int cf_n = 1;
    int *h_current_frontier = new int[V];
    h_current_frontier[0] = Source_node;
    // h_current_frontier[1] = 1;
    // h_current_frontier[2] = 4;

    int h_degree_array[V];

    int *h_outgoing_frontier = new int[E];

    // ---------------- GPU ----------------
    int *d_col;
    int *d_row_indices;
    int *d_current_frontier;
    int *d_degree_array;
    int *d_outgoing_frontier;

    cudaMalloc(&d_col, E * sizeof(int));
    cudaMalloc(&d_row_indices, (V) * sizeof(int));
    cudaMalloc(&d_current_frontier, V * sizeof(int));
    cudaMalloc(&d_degree_array, V * sizeof(int));
    cudaMalloc(&d_outgoing_frontier, E * sizeof(int));

    cudaMemcpy(d_col,
               h_col,
               E * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_row_indices,
               h_row_indices,
               (V) * sizeof(int),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_current_frontier,
               h_current_frontier,
               cf_n * sizeof(int),
               cudaMemcpyHostToDevice);

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

    // 1. Degree kernel

    int blocks = (cf_n + 255) / 256;
   clock_t starttt = clock();
    get_no_edges<<<blocks, 256>>>(
        d_current_frontier,
        d_row_indices,
        d_degree_array,
        cf_n);

    cudaDeviceSynchronize();
    // 2. Prefix scan + total edges
    int number_of_edges =
        getNumberEdges(
            d_degree_array,
            temp_storage_bytes,
            d_temp_storage,
            cf_n);


    int threadsPerBlock = 256;
    int edgesPerThread = 32;

    int totalThreads =
        (number_of_edges + edgesPerThread - 1) /
        edgesPerThread;

    int grid =
        (totalThreads + threadsPerBlock - 1) /
        threadsPerBlock;

    Advance_push<<<grid, threadsPerBlock>>>(
        d_current_frontier,
        d_outgoing_frontier,
        d_col,
        d_row_indices,
        d_degree_array, // now contains prefix sum
        number_of_edges,
        totalThreads,
        cf_n, number_of_edges);

    cudaDeviceSynchronize();
    clock_t enddd = clock();
    printf("Time Taken (GPU): %f ms\n", (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);

    //==========================================================
    // Copy back for testing
    //==========================================================

    cudaMemcpy(h_outgoing_frontier,
               d_outgoing_frontier,
               number_of_edges * sizeof(int),
               cudaMemcpyDeviceToHost);

    cout << "Outgoing Frontier:\n";

    for (int i = 0; i < number_of_edges; i++)
        cout << h_outgoing_frontier[i] << " ";

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