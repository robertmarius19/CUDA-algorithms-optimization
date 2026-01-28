#include <iostream>
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) \
  if((x) != cudaSuccess) { \
    std::cerr << "CUDA Error: " << cudaGetErrorString(x) << " at line " << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); \
  }

//constant memory for kernel
#define MAX_K 15 
__constant__ float c_kernel[MAX_K * MAX_K];

__global__ void conv2d_constant_shared(const float* __restrict__ input,
                                     int H, int W, int K,
                                     float* __restrict__ output,
                                     int BLOCK)
{
    // dynamic shared memory allocation
    extern __shared__ float shmem[];
    float* tile = shmem;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tileW = BLOCK + K - 1;
    int pad = K / 2;

    // Halo Loading
    for (int i = ty; i < tileW; i += BLOCK) {
        for (int j = tx; j < tileW; j += BLOCK) {
            int r = blockIdx.y * BLOCK + i - pad;
            int c = blockIdx.x * BLOCK + j - pad;
            
            float val = 0.0f;
            if (r >= 0 && r < H && c >= 0 && c < W) {
                val = input[r * W + c];
            }
            tile[i * tileW + j] = val;
        }
    }

    __syncthreads();

    int row = blockIdx.y * BLOCK + ty;
    int col = blockIdx.x * BLOCK + tx;

    if (row < H && col < W) {
        float sum = 0.0f;
        for (int ki = 0; ki < K; ++ki) {

            int tile_idx_base = (ty + ki) * tileW + tx;
            int kernel_idx_base = ki * K;
            
            #pragma unroll
            for (int kj = 0; kj < K; ++kj) {
                sum += tile[tile_idx_base + kj] * c_kernel[kernel_idx_base + kj];
            }
        }
        output[row * W + col] = sum;
    }
}

int main(int argc, char** argv) {
    int H = 1024, W = 1024, K = 7, BLOCK = 16;
    if (argc >= 2) H = std::atoi(argv[1]);
    if (argc >= 3) W = std::atoi(argv[2]);
    if (argc >= 4) K = std::atoi(argv[3]);

    if (K > MAX_K) {
        std::cerr << "Error: K is larger than MAX_K (" << MAX_K << ")\n";
        return 1;
    }

    size_t img_bytes = H * W * sizeof(float);
    size_t ker_bytes = K * K * sizeof(float);

    std::vector<float> h_input(H*W, 1.0f);
    std::vector<float> h_kernel(K*K, 1.0f/(K*K)); 
    std::vector<float> h_output(H*W, 0.0f);

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, img_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, img_bytes));

    CHECK_CUDA(cudaMemcpy(d_in, h_input.data(), img_bytes, cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpyToSymbol(c_kernel, h_kernel.data(), ker_bytes));

    dim3 block(BLOCK, BLOCK);
    dim3 grid((W + BLOCK - 1) / BLOCK, (H + BLOCK - 1) / BLOCK);

    int tileW = BLOCK + K - 1;
    size_t shared_mem_bytes = tileW * tileW * sizeof(float);

    conv2d_constant_shared<<<grid, block, shared_mem_bytes>>>(d_in, H, W, K, d_out, BLOCK);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    conv2d_constant_shared<<<grid, block, shared_mem_bytes>>>(d_in, H, W, K, d_out, BLOCK);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << ms << std::endl;

    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}