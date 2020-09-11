#include <stdio.h>
#include <cuda.h>
#include <cuda_profiler_api.h>

// Problem parameters. Should be templated, or possibly set at runtime.
#define N 16
#define M 8
#define K (N/M)
#define T 64
#define M3 (M*M*M)
#define N3 ((int64_t)N*N*N)
#define K3 (K*K*K)

#define cudaCheckErrors(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Apply one source pencil to all sinks
// Cells should probably already be in block order, i.e. (X,Y,Z,x,y,z)
template<typename FLOAT>
__global__ void block_on_block(FLOAT *cells, FLOAT *partial_sums, int source_pencil){
    int tid = threadIdx.x;
    int sinkblock = blockIdx.x;  // global 1D block index
    int sourcez = blockIdx.y;    // z offset of source block in source pencil
    int sourceblock = source_pencil*K + sourcez;  // global 1D
    
    FLOAT *sinks = cells + M3*sinkblock;
    FLOAT *sources = cells + M3*sourceblock;
    FLOAT *partials = partial_sums + N3*sourcez + M3*sinkblock;  // partials for this K,cell
    
    __shared__ FLOAT source_cache[T];
    // Sink loop
    for(int i = tid; i < M3; i += T){
        FLOAT thissink = sinks[i];
        FLOAT res = 0;
        
        // Source loop
        for(int j = tid; j < M3; j += T){
            // Each thread loads one source
            source_cache[tid] = sources[j];
            __syncthreads();

            // Each thread loops over all sources
            for(int j = 0; j < T; j++){
                // A dummy function, just trying to force some floating point math
                res += 2*((2*thissink + 1) * (2*source_cache[j] + 1));
            }
            // We change the source cache at the top of the loop; sync here
            __syncthreads();
        }
        
        partials[i] = res;
        __syncthreads();
    }
}

using FLOAT = float;

// Host driver
int main(int argc, char **argv){

    // Allocate and fill the cells
    FLOAT *cells = new FLOAT[N3];
    FLOAT *result = new FLOAT[N3];
    for(int64_t i = 0; i < N3; i++){
        cells[i] = 1.;
        result[i] = 0;
    }
    
    FLOAT **partials = new FLOAT*[K*K];
    for(int i = 0; i < K*K; i++){
        partials[i] = new FLOAT[N3*K];
    }
    
    cudaProfilerStart();
    
    // the device-side arrays
    FLOAT *dev_cells, *dev_partial_sums;
    cudaCheckErrors(cudaMalloc(&dev_cells, sizeof(FLOAT)*N3));
    // we're launching as pencil on cube, so we need a pencil's worth of partials for each cell
    cudaCheckErrors(cudaMalloc(&dev_partial_sums, sizeof(FLOAT)*N3*K));
    
    // send over the cells
    cudaCheckErrors(cudaMemcpy(dev_cells, cells, sizeof(FLOAT)*N3, cudaMemcpyHostToDevice));
    
    for(int i = 0; i < K*K; i++){
        dim3 grid(K3,K);
        dim3 block(T);
        block_on_block<<<grid,block>>>(dev_cells, dev_partial_sums, i);
        
        // Get the result
        cudaCheckErrors(cudaMemcpy(partials[i], dev_partial_sums, sizeof(FLOAT)*N3*K, cudaMemcpyDeviceToHost));
        cudaCheckErrors(cudaDeviceSynchronize());
    }
    
    cudaProfilerStop();
    
    // Now sum the K^3 results per cell
    for(int64_t i = 0; i < N3; i++){
        for(int64_t j = 0; j < K*K; j++){
            for(int64_t kk = 0; kk < K; kk++){
                //printf("partials[j][kk*N3 + i] = %g\n", partials[j][kk*N3 + i]);
                result[i] += partials[j][kk*N3 + i];
            }
        }
    }
    
    // Verify result
    printf("result[0] = %g\n", result[0]);
    int i = 0;
    for(i = 0; i < N3; i++){
        if (result[i] != 18.*N3){
            printf("result[%d] = %g\n", i, result[i]);
            break;
        }
    }
    if(i == N3)
        printf("Verified!\n");
    
    cudaCheckErrors(cudaFree(dev_cells));
    cudaCheckErrors(cudaFree(dev_partial_sums));
    
    for(int i = 0; i < K*K; i++)
        delete[] partials[i];
    delete[] partials;
    
    delete[] cells;
    
    return 0;
}
