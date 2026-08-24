/*
THE INPUT OF THE GRAPH WILL BE IN CSR FORMAT.
*/
#include <time.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <bits/stdc++.h>
#include <iostream>
#include <cub/cub.cuh> // -> for prefix sum

#define EDGESPERTHREAD 2
#define THREADSPERBLOCK 256
#define INF INT_MAX

// Hyperparameter for Batch Size
#define K 4 

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
{ // N -> number of meta-vertices in the unified "current frontier"
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        int meta_v = current_frontier[idx];
        int v = meta_v % V; 
        degree_array[idx] = degree(row_indices, v, V, E);
    }
}

__global__ void initZero(int *array, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        array[tid] = 0;
    }
}

__global__ void generate_frontier(int *vkeep, int *prefix_vkeep, int KV, int *unvisited_frontier)
{
    // launch up to K * V 
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < KV)
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
{ 
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
}

//========================================================================================================
// PUSH ADVANCE KERNEL
//========================================================================================================

__global__ void Advance_push(int *current_frontier, int *bitmask,
                             int *col, int *row_indices, int *weights,
                             int *prefix_sum, int edges, int *minDist,
                             int nothreads, int cf_n, int V) // N == current frontier size.
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < nothreads)
    {
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
        
        int meta_parent = current_frontier[bstart];
        int parent = meta_parent % V;
        int batch_k = meta_parent / V;
        
        int colIndex = row_indices[parent] + offset;
        int counter = 0;
        
        for (int i = start; i < end; i++)
        {
            int v_neighbor = col[colIndex];
            int meta_col = batch_k * V + v_neighbor; 
            
            int newDist = minDist[meta_parent] + weights[colIndex];
            int old = atomicMin(&minDist[meta_col], newDist);
            
            if (newDist < old)
                bitmask[meta_col] = 1; 
                
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
                    
                // Decode NEXT Meta-Node
                meta_parent = current_frontier[bstart];
                parent = meta_parent % V;
                batch_k = meta_parent / V;
                
                colIndex = row_indices[parent];
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
    if (N <= 0) return 0;
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
    Advance_push<<<grid, THREADSPERBLOCK>>>(d_current_frontier, vkeep, d_col, d_row_indices, d_weights, d_degree_array, number_of_edges, d_minDist, totalThreads, cf_n, V);
    cf_n = getNumberEdges(vkeep, prefix_vkeep, temp_storage_bytes, d_temp_storage, K * V);
    grid = (K * V + THREADSPERBLOCK - 1) / THREADSPERBLOCK;
    generate_frontier<<<grid, THREADSPERBLOCK>>>(vkeep, prefix_vkeep, K * V, d_current_frontier);
    initZero<<<grid, THREADSPERBLOCK>>>(vkeep, K * V);
    cudaDeviceSynchronize();

    return cf_n;
}

void cudaSSSP(int *d_current_frontier, int *d_col, int *d_row_indices, int *d_weights, void *d_temp_storage,
              size_t temp_storage_bytes, int *d_minDist, int *prefix_vkeep, int *d_degree_array, int *vkeep, int V, int E)
{
    int cf_n = K; // Initially K sources in our unified frontier 
    while(cf_n > 0){
        cf_n = cudaSSSP_push(d_current_frontier, d_col, d_row_indices, d_weights, d_temp_storage, temp_storage_bytes, d_minDist, prefix_vkeep, d_degree_array, vkeep, cf_n, V, E); 
    }
}

int main()
{
    int E, V, Source_node;
    cin >> E >> V >> Source_node;
    
    // ---------------- CPU ----------------
    int *h_col = new int[E];
    int *h_weight = new int[E];
    int *h_row_indices = new int[V];
    
    // Scale state arrays by K
    int *h_distance = new int[K * V];
    int *h_current_frontier = new int[K * V];
    
    fill(h_distance, h_distance + (K * V), INF);
    
    input_array(h_col, E);
    input_array(h_weight, E);
    input_array(h_row_indices, V);

    // Initial unified frontier includes all K sources 
    int cf_n = K;
    for (int k = 0; k < K; k++) {
        // Using `(Source_node + k)%V` for example purpose... 
        int batch_source = (Source_node + k) % V; 
        
        // distance[k * V + sourceNode] = 0
        h_current_frontier[k] = k * V + batch_source;
        h_distance[k * V + batch_source] = 0; 
    }

    // ---------------- GPU ----------------
    int *d_col;
    int *d_weight;
    int *d_row_indices;
    
    // Scale these allocations by K
    int *d_current_frontier;
    int *d_degree_array;
    int *d_vkeep;
    int *d_vprekeep;
    int *d_distance;

    // O(6*K*V+2E) -> space complexity on DRAM.


    cudaMalloc(&d_row_indices, V * sizeof(int)); 
    cudaMalloc(&d_col, E * sizeof(int));
    cudaMalloc(&d_weight, E * sizeof(int));
    cudaMalloc(&d_distance, K * V * sizeof(int));
    cudaMalloc(&d_current_frontier, K * V * sizeof(int));
    cudaMalloc(&d_degree_array, K * V * sizeof(int));
    cudaMalloc(&d_vkeep, K * V * sizeof(int));
    cudaMalloc(&d_vprekeep, K * V * sizeof(int));

    cudaMemset(d_degree_array, 0, K * V * sizeof(int));
    cudaMemset(d_vkeep, 0, K * V * sizeof(int));
    cudaMemset(d_vprekeep, 0, K * V * sizeof(int));

    // Topology arrays are copied normally (size V / E)
    cudaMemcpy(d_col, h_col, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_indices, h_row_indices, V * sizeof(int), cudaMemcpyHostToDevice);
    
    // States arrays are copied scaled to K * V
    cudaMemcpy(d_distance, h_distance, (K * V) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_current_frontier, h_current_frontier, cf_n * sizeof(int), cudaMemcpyHostToDevice);

    //==========================================================
    // Allocate CUB temporary storage ONCE
    //==========================================================
    size_t temp_storage_bytes = 0;
    
    // The maximum elements we scan over is K * V (for vkeep generation and prefix sums)
    int max_scan_size = K * V;

    cub::DeviceScan::ExclusiveSum(
        nullptr,
        temp_storage_bytes,
        d_vkeep,
        d_vprekeep,
        max_scan_size);

    void *d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    //==========================================================
    // SSSP
    //==========================================================
    
    clock_t starttt = clock();
    cudaSSSP(d_current_frontier, d_col, d_row_indices, d_weight, d_temp_storage, temp_storage_bytes, d_distance, d_vprekeep, d_degree_array, d_vkeep, V, E); 
    clock_t enddd = clock();

    //==========================================================
    // Copy back for testing
    //==========================================================

    cudaMemcpy(h_distance, d_distance, K * V * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "######### " << K << "x BATCHED SSSP GUNROCK ########" << endl;
    cout << "vertices           : " << V << '\n';
    cout << "edges              : " << E << '\n';
    printf("Total Time Taken (GPU) : %f ms\n", (((double)enddd - (double)starttt) / CLOCKS_PER_SEC) * 1000);

    // Check results independently across K batches
    for(int k = 0; k < K; k++){
        int maxi = INT_MIN;
        for (int i = 0; i < V; i++)
        {
            if(h_distance[k * V + i] != INF)
                maxi = max(maxi, h_distance[k * V + i]);
        }
        cout << "[Batch " << k << "] -> Source node: " << (Source_node + k) % V 
             << " | MAX Distance: " << maxi << endl;
    }

    //==========================================================
    // Cleanup
    //==========================================================

    cudaFree(d_row_indices);
    cudaFree(d_col);
    cudaFree(d_weight);
    cudaFree(d_distance);
    cudaFree(d_current_frontier);
    cudaFree(d_degree_array);
    cudaFree(d_vkeep);
    cudaFree(d_vprekeep);
    cudaFree(d_temp_storage);

    delete[] h_col;
    delete[] h_weight;
    delete[] h_row_indices;
    delete[] h_current_frontier;
    delete[] h_distance;

    return 0;
}