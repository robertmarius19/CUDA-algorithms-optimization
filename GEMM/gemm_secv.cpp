#include <bits/stdc++.h>

int main(int argc, char** argv) {
    int N = 4096;
    if (argc >= 2) N = std::atoi(argv[1]);
    std::vector<float> A(N*N, 1.0f);
    std::vector<float> B(N*N, 1.0f);
    std::vector<float> C(N*N, 0.0f);

  
    auto t0 = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  
    double total = 0.0;
    for(auto x : C) total += x;

    std::cout << elapsed_ms << std::endl;

    return 0;
}
