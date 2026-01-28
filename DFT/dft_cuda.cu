#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) \
    if((x) != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(x) \
                  << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define BLOCK 256
#define PI 3.14159265358979323846f

// Each thread computes one output frequency k
__global__ void dft_cuda_shared(const float* __restrict__ in_real,
                                const float* __restrict__ in_imag,
                                float* __restrict__ out_real,
                                float* __restrict__ out_imag,
                                int N)
{
    __shared__ float sh_real[BLOCK];
    __shared__ float sh_imag[BLOCK];

    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    float sum_real = 0.0f;
    float sum_imag = 0.0f;

    float angle_base = -2.0f * PI * k / N;

    for (int tile = 0; tile < N; tile += BLOCK) {

        int idx = tile + threadIdx.x;
        if (idx < N) {
            sh_real[threadIdx.x] = in_real[idx];
            sh_imag[threadIdx.x] = in_imag[idx];
        } else {
            sh_real[threadIdx.x] = 0.0f;
            sh_imag[threadIdx.x] = 0.0f;
        }

        __syncthreads();

        int tileSize = min(BLOCK, N - tile);
        for (int n = 0; n < tileSize; n++) {
            float angle = angle_base * (tile + n);
            float s, c;
            sincosf(angle, &s, &c);

            sum_real += sh_real[n] * c - sh_imag[n] * s;
            sum_imag += sh_real[n] * s + sh_imag[n] * c;
        }

        __syncthreads();
    }

    out_real[k] = sum_real;
    out_imag[k] = sum_imag;
}
int main(int argc, char** argv) {
    int N = 4096;
    if (argc >= 2) N = std::atoi(argv[1]);
    size_t bytes = N * sizeof(float);

    std::vector<float> h_real(N, 1.0f);
    std::vector<float> h_imag(N, 0.0f);
    std::vector<float> h_out_real(N), h_out_imag(N);

    float *d_real, *d_imag, *d_out_real, *d_out_imag;
    CHECK_CUDA(cudaMalloc(&d_real, bytes));
    CHECK_CUDA(cudaMalloc(&d_imag, bytes));
    CHECK_CUDA(cudaMalloc(&d_out_real, bytes));
    CHECK_CUDA(cudaMalloc(&d_out_imag, bytes));

    CHECK_CUDA(cudaMemcpy(d_real, h_real.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_imag, h_imag.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dft_cuda_shared<<<grid, block>>>(d_real, d_imag, d_out_real, d_out_imag, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CHECK_CUDA(cudaMemcpy(h_out_real.data(), d_out_real, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_out_imag.data(), d_out_imag, bytes, cudaMemcpyDeviceToHost));

    double energy = 0.0;
    for (int i = 0; i < N; i++)
        energy += h_out_real[i] * h_out_real[i] + h_out_imag[i] * h_out_imag[i];

    std::cout << ms << std::endl;

    cudaFree(d_real);
    cudaFree(d_imag);
    cudaFree(d_out_real);
    cudaFree(d_out_imag);
}
