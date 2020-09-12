#include <stdio.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <assert.h>
using namespace std::chrono; 

// Problem parameters. Should be templated, or possibly set at runtime.
#define N 256
#define M 32
#define K (N/M)
#define T 64
#define M3 (M*M*M)
#define N3 ((int64_t)N*N*N)
#define K3 (K*K*K)

#define PAIR_OP(sink,source) 2*((2*sink + 1) * (2*source + 1))

/*

TODO
- Multiple streams
- Pinned memory
- Real radiative transfer function
- Could consider pencil-on-block or pencil-on-pencil if CPU memory is getting out of hand.
*/

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
                res += PAIR_OP(thissink, source_cache[j]);
            }
            // We change the source cache at the top of the loop; sync here
            __syncthreads();
        }
        
        partials[i] = res;
        __syncthreads();
    }
}

using FLOAT = float;

void do_cpu(FLOAT *cells, FLOAT *result){
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < N3; i++){
        result[i] = 0;
        for(int j = 0; j < N3; j++){
            result[i] += PAIR_OP(cells[i], cells[j]);
        }
    }
}

void check_cpu(FLOAT *cells, FLOAT *gpu_result){
    FLOAT *cpu_result = new FLOAT[N3];
    
    auto start = high_resolution_clock::now(); 
    do_cpu(cells, cpu_result);
    auto elapsed = duration_cast<nanoseconds>(high_resolution_clock::now() - start);
    printf("CPU execution took %.3g seconds\n", elapsed.count()/1e9);
    
    for(int i = 0; i < N3; i++){
        FLOAT rerr = std::abs((cpu_result[i] - gpu_result[i])/cpu_result[i]);
        if(rerr > 1e-4){
            printf("Error! cpu_result[%d] = %g, gpu_result[%d] = %g, rerr = %g\n", i, cpu_result[i], i, gpu_result[i], rerr);
            break;
        }
        
        if(i == N3-1)
            printf("GPU result matches CPU result!\n");
    }
    
    delete[] cpu_result;
}

// Host driver
int main(int argc, char **argv){
    // check params
    assert(M*K == N);
    assert((M3/T)*T == M3);
    assert((T/32)*32 == T);

    // Allocate and fill the cells
    FLOAT *cells = new FLOAT[N3];
    FLOAT *result = new FLOAT[N3];
    for(int64_t i = 0; i < N3; i++){
        cells[i] = ((double) rand()) / RAND_MAX;
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
    auto start = high_resolution_clock::now(); 
    #pragma omp parallel for schedule(static)
    for(int64_t i = 0; i < N3; i++){
        for(int64_t j = 0; j < K*K; j++){
            for(int64_t kk = 0; kk < K; kk++){
                //printf("partials[j][kk*N3 + i] = %g\n", partials[j][kk*N3 + i]);
                result[i] += partials[j][kk*N3 + i];
            }
        }
    }
    auto elapsed = duration_cast<nanoseconds>(high_resolution_clock::now() - start);
    printf("Reduction took %.3g seconds\n", elapsed.count()/1e9);
    
    // Anything much bigger than this is unlikely to complete on the CPU in a useful amount of time
    if(N <= 64)
        check_cpu(cells, result);
    
    // Verify result, only useful if the cells are filled with a constant
    /*printf("result[0] = %g\n", result[0]);
    for(int i = 0; i < N3; i++){
        if (result[i] != 18.*N3){
            printf("result[%d] = %g\n", i, result[i]);
            break;
        }
        if(i == N3-1)
            printf("Verified!\n");
    }
    */
    
    cudaCheckErrors(cudaFree(dev_cells));
    cudaCheckErrors(cudaFree(dev_partial_sums));
    
    for(int i = 0; i < K*K; i++)
        delete[] partials[i];
    delete[] partials;
    
    delete[] cells;
    
    return 0;
}
