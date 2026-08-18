/*
THE INPUT OF THE GRAPH WILL BE IN CSR FORMAT. THERE WILL BE A HELPER KERNEL TO CONVERT IT TO CSC AND BITMAP
FOR PULL MODE.
*/
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <bits/stdc++.h>
#include <iostream>
#include <cub/cub.cuh> // -> for prefix sum
#define EDGESPERTHREAD 2
#define THREADSPERBLOCK 256
#define INF INT_MAX
using namespace std;

//========================================================================================================
// Helper Functions GPU
//========================================================================================================
__device__ int degree(int *row_offsets, int vertex, int V, int E)
{
    int sub = (vertex == V - 1) ? E : row_offsets[vertex + 1];
    return sub - row_offsets[vertex];
}
//========================================================================================================
// Helper Kernels GPU
//========================================================================================================
__global__ void getDegree(int *current_frontier, int *row_indices, int *degree_array, int N, int V, int E)
{ //-> number of vertex in "current fronteir" (N)
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        degree_array[idx] = degree(row_indices, current_frontier[idx], V, E);
}


__global__ void initZero(int *array, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        array[tid] = 0;
    }
}

__global__ void generate_frontier(int *vkeep, int *prefix_vkeep, int V, int *unvisited_frontier)
{
    // launch V
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < V)
    {
        if (vkeep[idx])
        {
            unvisited_frontier[prefix_vkeep[idx]] = idx;
        }
    }
}

//========================================================================================================
// HELPER CUB FUNCTIONS
//========================================================================================================

void cubExclusiveScan(int *d_in, int *d_out, size_t temp_storage_bytes, void *d_temp_storage, int N)
{ // N-> size of frontier.

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    cudaDeviceSynchronize();
}
//========================================================================================================
// PUSH ADVANCE KERNEL
//========================================================================================================

__global__ void Advance_push(int *current_frontier, int *bitmask,
                             int *col, int *row_indices, int *weights,
                             int *prefix_sum, int edges, int *minDist,
                             int nothreads, int cf_n) // N == current frontier size.
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nothreads)
    {
        // Advance_push — same fix, using edges/nothreads
        // Potential BUGS: integer Overflow, weight index, a few edges cases(simple ones!!).
        int start = (idx == 0) ? 0 : (int)floor(idx * ((double)edges / nothreads));
        int end = (idx == nothreads - 1) ? edges : (int)floor((idx + 1) * ((double)edges / nothreads));
        int bstart = 0;
        int benf = cf_n - 1;
        while (benf > bstart)
        {
            int mid = (bstart + benf + 1) / 2;
            if (prefix_sum[mid] <= start)
                bstart = mid;
            else
                benf = mid - 1;
        }

        int offset = start - prefix_sum[bstart];
        int next_ps = (bstart + 1 < cf_n) ? prefix_sum[bstart + 1] : edges;
        int degree = next_ps - prefix_sum[bstart];
        int colIndex = row_indices[current_frontier[bstart]] + offset;
        int parent = current_frontier[bstart];
        int counter = 0;
        for (int i = start; i < end; i++)
        {
            int newDist = minDist[parent] + weights[colIndex];
            int old = atomicMin(&minDist[col[colIndex]], newDist);
            if (newDist < old)
                bitmask[col[colIndex]] = 1;
            colIndex++;
            counter++;
            if (counter >= degree - offset)
            {
                bstart++;
                while (bstart < cf_n)
                {
                    int np = (bstart + 1 < cf_n) ? prefix_sum[bstart + 1] : edges;
                    degree = np - prefix_sum[bstart];
                    if (degree > 0)
                        break;
                    bstart++;
                }
                if (bstart >= cf_n)
                    break;
                colIndex = row_indices[current_frontier[bstart]];
                parent = current_frontier[bstart];
                counter = 0;
                offset = 0;
            }
        }
    }
}
//========================================================================================================
// Helper Functions CPU
//========================================================================================================
int getNumberEdges(int *prefix_sum_in, int *prefix_sum_out, size_t temp_storage_bytes, void *d_temp_storage, int N)
{
    if (N <= 0)
        return 0;
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
//========================================================================================================
// Wrapper Functions CPU
//========================================================================================================
int cudaSSSP_push(int *d_current_frontier, int *d_col, int *d_row_indices, int *d_weights, void *d_temp_storage,
                  size_t temp_storage_bytes, int *d_minDist, int *prefix_vkeep, int *d_degree_array, int *vkeep, int cf_n, int V, int E)
{
    int blocks = (cf_n + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    getDegree<<<blocks, THREADSPERBLOCK>>>(d_current_frontier, d_row_indices, d_degree_array, cf_n, V, E);
    int number_of_edges = getNumberEdges(d_degree_array, d_degree_array, temp_storage_bytes, d_temp_storage, cf_n);
    int totalThreads = (number_of_edges + EDGESPERTHREAD - 1) / EDGESPERTHREAD;
    int grid = (totalThreads + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    Advance_push<<<grid, THREADSPERBLOCK>>>(d_current_frontier, vkeep, d_col, d_row_indices, d_weights, d_degree_array, number_of_edges, d_minDist, totalThreads, cf_n);
    cf_n = getNumberEdges(vkeep, prefix_vkeep, temp_storage_bytes, d_temp_storage, V);
    grid = (V + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    generate_frontier<<<grid, THREADSPERBLOCK>>>(vkeep, prefix_vkeep, V, d_current_frontier);
    initZero<<<grid, THREADSPERBLOCK>>>(vkeep, V);
    cudaDeviceSynchronize();

    return cf_n;
}
void cudaSSSP(int *d_current_frontier, int *d_col, int *d_row_indices, int *d_weights, void *d_temp_storage,
size_t temp_storage_bytes, int *d_minDist, int *prefix_vkeep, int *d_degree_array, int *vkeep, int V, int E, int Source)
{
    int cf_n = 1 ; 
    while(cf_n>0){
        cf_n = cudaSSSP_push(d_current_frontier,d_col,d_row_indices,d_weights,d_temp_storage,temp_storage_bytes,d_minDist,prefix_vkeep,d_degree_array,vkeep,cf_n,V,E); 
    }
}
int main()
{
    int E, V, Source_node;
    cin >> E >> V >> Source_node;
    // ---------------- CPU ----------------
    int *h_col = new int[E];
    int *h_weight = new int[E];
    int *h_distance = new int[V];
    int *h_row_indices = new int[V];
    int *h_current_frontier = new int[V];
    // memset(h_distance, INF, V * sizeof(int));
    fill(h_distance, h_distance + V, INF);
    input_array(h_col, E);
    input_array(h_weight, E);
    input_array(h_row_indices, V);

    // Initial frontier = source
    int cf_n = 1;
    h_current_frontier[0] = Source_node;
    h_distance[Source_node] = 0;

    // ---------------- GPU ----------------
    int *d_col;
    int *d_weight;
    int *d_row_indices;
    int *d_current_frontier;
    int *d_degree_array;
    int *d_vkeep;
    int *d_vprekeep;
    int *d_distance;

    // O(6V+2E) -> space complexity on DRAM.

    cudaMalloc(&d_row_indices, V * sizeof(int)); 
    cudaMalloc(&d_col, E * sizeof(int));
    cudaMalloc(&d_weight, E * sizeof(int));
    cudaMalloc(&d_distance, V * sizeof(int));
    cudaMalloc(&d_current_frontier, V * sizeof(int));
    cudaMalloc(&d_degree_array, V * sizeof(int));
    cudaMalloc(&d_vkeep, V * sizeof(int));
    cudaMalloc(&d_vprekeep, V * sizeof(int));

    cudaMemset(d_degree_array, 0, V * sizeof(int));
    cudaMemset(d_vkeep, 0, V * sizeof(int));
    cudaMemset(d_vprekeep, 0, V * sizeof(int));

    cudaMemcpy(d_col, h_col, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, h_row_indices, (V) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance, h_distance, (V) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_current_frontier, h_current_frontier, cf_n * sizeof(int), cudaMemcpyHostToDevice);

    //==========================================================
    // Allocate CUB temporary storage ONCE
    //==========================================================

    size_t temp_storage_bytes = 0;

    int max_scan_size = (V > E) ? V : E;

    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_storage_bytes,
        d_vkeep, // using d_keep just to provide a valid pointer
        d_vprekeep,
        max_scan_size);

    void *d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    //==========================================================
    // SSSP
    //==========================================================
    
    clock_t starttt = clock();
    cudaSSSP(d_current_frontier,d_col,d_row_indices, d_weight, d_temp_storage,temp_storage_bytes,d_distance, d_vprekeep,d_degree_array,d_vkeep,V,E,Source_node); 
    clock_t enddd = clock();

    //==========================================================
    // Copy back for testing
    //==========================================================

    cudaMemcpy(h_distance,
               d_distance,
               V * sizeof(int),
               cudaMemcpyDeviceToHost);

    cout << "Distance Array 1st 20:\n";

    for (int i = 0; i < 5; i++)
        cout << h_distance[i] << " ";

    // cout << "\nDistance Array last 20:\n";

    // for (int i = V - 21; i < V; i++)
    //     cout << h_distance[i] << " ";
    int maxi = INT_MIN;
    for (int i = 0; i < V; i++)
    {
        if(h_distance[i]!=INT_MAX)
        maxi = max(maxi, h_distance[i]);
    }
    cout << endl;
    cout << "#########SSSP GUNROCK########" << endl;
    cout << "vertices           : " << V << '\n';
    cout << "edges              : " << E << '\n';
    cout << "source             : " << Source_node << endl;
    cout << "MAX Distance       : " << maxi << endl;
    printf("Time Taken (GPU)   : %f ms\n", (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);

    //==========================================================
    // Cleanup
    //==========================================================

    cudaFree(d_col);
    cudaFree(d_row_indices);
    cudaFree(d_current_frontier);
    cudaFree(d_degree_array);
    cudaFree(d_temp_storage);

    delete[] h_col;
    delete[] h_row_indices;
    delete[] h_current_frontier;

    return 0;
}
