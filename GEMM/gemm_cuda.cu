#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK 32

#define CUDA_CHECK(x) \
    if((x) != cudaSuccess) { \
        std::cout << "CUDA Error: " << cudaGetErrorString(x) << " at line " << __LINE__ << std::endl; exit(1); \
    }

#define CUBLAS_CHECK(x) \
    if((x) != CUBLAS_STATUS_SUCCESS) { \
        std::cout << "cuBLAS Error at line " << __LINE__ << std::endl; exit(1); \
    }

// Shared-memory GEMM kernel
__global__ void gemm_shared(const float* A, const float* B, float* C, int N) {
    __shared__ float As[BLOCK][BLOCK];
    __shared__ float Bs[BLOCK][BLOCK];

    int row = blockIdx.y * BLOCK + threadIdx.y;
    int col = blockIdx.x * BLOCK + threadIdx.x;

    float sum = 0.0f;

    for(int t = 0; t < N; t += BLOCK) {
        if(row < N && t + threadIdx.x < N)
            As[threadIdx.y][threadIdx.x] = A[row*N + t + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if(col < N && t + threadIdx.y < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y)*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for(int k = 0; k < BLOCK; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if(row < N && col < N)
        C[row*N + col] = sum;
}

int main(int argc, char** argv) {
    int N = 4096;
    if (argc >= 2) N = std::atoi(argv[1]);

    size_t bytes = N*N*sizeof(float);
    std::vector<float> hA(N*N, 1.0f), hB(N*N, 1.0f), hC(N*N, 0.0f);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    float ms = 0.0f;

    if(N < 2048) {
        
        dim3 threads(BLOCK,BLOCK);
        dim3 blocks((N+BLOCK-1)/BLOCK, (N+BLOCK-1)/BLOCK);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        gemm_shared<<<blocks, threads>>>(dA,dB,dC,N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms,start,stop);
    } else {
        
        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        float alpha = 1.0f;
        float beta  = 0.0f;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        
        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            dB, N,
            dA, N,
            &beta,
            dC, N));

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms,start,stop);

        cublasDestroy(handle);
    }

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));

    double sum = 0.0;
    for(float x : hC) sum += x;

    std::cout << ms << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}
