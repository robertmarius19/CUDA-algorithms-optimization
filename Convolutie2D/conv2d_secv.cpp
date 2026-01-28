#include <iostream>
#include <vector>
#include <chrono>


std::vector<float> pad_image(const std::vector<float>& input, int H, int W, int pad) {
    int padded_H = H + 2 * pad;
    int padded_W = W + 2 * pad;
    std::vector<float> padded(padded_H * padded_W, 0.0f);

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            padded[(i + pad) * padded_W + (j + pad)] = input[i * W + j];
        }
    }
    return padded;
}

void conv2d_seq_optimized(const std::vector<float>& padded_input, int H, int W,
                          const std::vector<float>& kernel, int K,
                          std::vector<float>& output) {
    int pad = K / 2;
    int padded_W = W + 2 * pad;

    for (int i = 0; i < H; i++) {
        for (int j = 0; j < W; j++) {
            float sum = 0.0f;
         
            const float* input_ptr_base = &padded_input[i * padded_W + j];
            
            for (int ki = 0; ki < K; ki++) {
                const float* input_row = input_ptr_base + (ki * padded_W);
                const float* kernel_row = &kernel[ki * K];
                for (int kj = 0; kj < K; kj++) {
                    sum += input_row[kj] * kernel_row[kj];
                }
            }
            output[i * W + j] = sum;
        }
    }
}

int main(int argc, char** argv) {
    int N = 1024;
    if (argc >= 2) N = std::atoi(argv[1]);
    int H = N, W = N, K = 3;

    std::vector<float> input(H*W, 1.0f);
    std::vector<float> kernel(K*K, 1.0f/9.0f);
    std::vector<float> output(H*W, 0.0f);

  
    int pad = K / 2;
    std::vector<float> padded_input = pad_image(input, H, W, pad);

    auto t0 = std::chrono::high_resolution_clock::now();

    conv2d_seq_optimized(padded_input, H, W, kernel, K, output);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << ms << std::endl;
    return 0;
}