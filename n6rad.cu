#include <stdio.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <chrono>
#include <assert.h>
using namespace std::chrono; 

// Problem parameters
// TODO: Set at runtime (check performance)
#define N 256
#define M 32
#define K (N/M)
#define T 64
#define M3 (M*M*M)
#define N3 ((int64_t)N*N*N)
#define K3 (K*K*K)

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

template<typename FLOAT>
__host__ __device__ inline FLOAT pair_op(FLOAT sink, FLOAT source, int dist2){
    // faster to multiply by (1/r^2) than divide by r^2
    return 2*((2*sink + 1) * (2*source + 1))*(1/static_cast<FLOAT>(dist2));
}

// Take 1D index i
// and convert it to the corresponding 3D (ix,iy,iz) index
// expand lets you multiply the final result by some factor (useful for converting block indices to cell indices)
__host__ __device__ inline void unravel_index(int i, int dim, int &ix, int &iy, int &iz, int expand=1){
    ix = i / (dim*dim);
    iy = i / dim - ix*dim;
    iz = i % dim;
    
    ix *= expand;
    iy *= expand;
    iz *= expand;
}

// Apply one source block to one sink block
// Cells must be in block order, i.e. (X,Y,Z,x,y,z)
template<typename FLOAT>
__global__ void block_on_block(FLOAT *cells, FLOAT *partial_sums, int source_pencil){
    int tid = threadIdx.x;
    int sinkblock = blockIdx.x;  // global 1D block index
    int sourcez = blockIdx.y;    // z offset of source block in source pencil
    int sourceblock = source_pencil*K + sourcez;  // global 1D
    
    // Get the (i,j,k) index of the sink and source block in the unpermuted N^3 grid
    int _sinki, _sinkj, _sinkk;
    int _sourcei, _sourcej, _sourcek;
    unravel_index(sinkblock, K, _sinki, _sinkj, _sinkk, M);
    unravel_index(sourceblock, K, _sourcei, _sourcej, _sourcek, M);
    
    FLOAT *sinks = cells + M3*sinkblock;
    FLOAT *sources = cells + M3*sourceblock;
    FLOAT *partials = partial_sums + N3*sourcez + M3*sinkblock;  // partials for this K,cell
    
    __shared__ FLOAT source_cache[T];
    // Sink loop
    for(int i = tid; i < M3; i += T){
        FLOAT thissink = sinks[i];
        FLOAT res = 0;
        
        // Get the location within the block
        int sinki, sinkj, sinkk;
        unravel_index(i, M, sinki, sinkj, sinkk);
        
        // and add on the coordinates of the block
        sinki += _sinki;
        sinkj += _sinkj;
        sinkk += _sinkk;
        
        // Source loop
        for(int j = 0; j < M3; j += T){
            // Each thread loads one source
            source_cache[tid] = sources[j+tid];
            __syncthreads();

            // Each thread loops over all sources
            for(int t = 0; t < T; t++){
                // TODO: there's a few options to optimize this indexing math
                // - get rid of mod
                // - if T divides M^2, only have to update j,k
                // - can precompute index(es) into shared array
                // - could use bit shifts, but probably don't want to constrain to power of 2
                
                // Get the location within the block
                int sourcei, sourcej, sourcek;
                unravel_index(j+t, M, sourcei, sourcej, sourcek);

                // and add on the coordinates of the block
                sourcei += _sourcei;
                sourcej += _sourcej;
                sourcek += _sourcek;
                
                int dist2 = (sinki - sourcei)*(sinki - sourcei) + (sinkj - sourcej)*(sinkj - sourcej) + (sinkk - sourcek)*(sinkk - sourcek);
                
                // A dummy function, just trying to force some floating point math
                res += pair_op(thissink, source_cache[t], dist2);
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
    // Do the same calculation on the GPU
    // We'll assume the cells are not permuted into blocks; i.e. standard C row-major order
    // This will provide some insurance against repeated indexing errors
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < N3; i++){
        result[i] = 0;
        int ix, iy, iz;
        unravel_index(i, N, ix, iy, iz);
        for(int j = 0; j < N3; j++){
            int jx, jy, jz;
            unravel_index(j, N, jx, jy, jz);
            int dist2 = (ix - jx)*(ix - jx) + (iy - jy)*(iy - jy) + (iz - jz)*(iz - jz);
            result[i] += pair_op(cells[i], cells[j], dist2);
        }
    }
}

// The cells are laid out so that individual blocks are physically contiguous for the GPU
// Unpermute them into standard C order
void unpermute_cells(FLOAT *unpermuted_cells, FLOAT *cells){
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < N; i++){
        int ki = i / M;
        int mi = i % M;
        
        for(int j = 0; j < N; j++){
            int kj = j / M;
            int mj = j % M;
            
            for(int k = 0; k < N; k++){
                int kk = k / M;
                int mk = k % M;
                
                int bstart = (ki*K*K + kj*K + kk)*M3;
                int off = mi*M*M + mj*M + mk;
                int pi = bstart + off;
        
                int to = i*N*N + j*N + k;
                
                unpermuted_cells[to] = cells[pi];
            }
        }
    }
}

void check_cpu(FLOAT *cells, FLOAT *gpu_result){
    FLOAT *cpu_result = new FLOAT[N3];
    
    // Make an unpermuted version for the CPU to operate on
    FLOAT *unpermuted_cells = new FLOAT[N3];
    unpermute_cells(unpermuted_cells, cells);
    
    auto start = high_resolution_clock::now(); 
    do_cpu(unpermuted_cells, cpu_result);
    auto elapsed = duration_cast<nanoseconds>(high_resolution_clock::now() - start);
    printf("CPU execution took %.3g seconds\n", elapsed.count()/1e9);
    
    // Reorder the GPU result into the order that the CPU produces
    FLOAT *unpermuted_gpu_result = new FLOAT[N3];
    unpermute_cells(unpermuted_gpu_result, gpu_result);
    
    for(int i = 0; i < N3; i++){
        FLOAT rerr = std::abs((cpu_result[i] - unpermuted_gpu_result[i])/cpu_result[i]);
        if(rerr > 1e-4){
            printf("Error! cpu_result[%d] = %g, gpu_result[%d] = %g, rerr = %g\n", i, cpu_result[i], i, unpermuted_gpu_result[i], rerr);
            break;
        }
        
        if(i == N3-1)
            printf("GPU result matches CPU result!\n");
    }
    
    delete[] unpermuted_cells;
    delete[] unpermuted_gpu_result;
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
    auto start = high_resolution_clock::now();
    
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
    
    auto elapsed = duration_cast<nanoseconds>(high_resolution_clock::now() - start);
    printf("GPU took %.3g seconds for N=%d, M=%d\n", elapsed.count()/1e9, N, M);
    
    cudaProfilerStop();
    
    // Now sum the K^3 results per cell
    start = high_resolution_clock::now();
    #pragma omp parallel for schedule(static)
    for(int64_t i = 0; i < N3; i++){
        for(int64_t j = 0; j < K*K; j++){
            for(int64_t kk = 0; kk < K; kk++){
                result[i] += partials[j][kk*N3 + i];
            }
        }
    }
    elapsed = duration_cast<nanoseconds>(high_resolution_clock::now() - start);
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
