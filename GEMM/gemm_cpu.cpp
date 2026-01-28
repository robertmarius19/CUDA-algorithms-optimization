#include <iostream>
#include <cblas.h>
#include <vector>
#include <chrono>

int main(int argc, char** argv) {
    int N = 4096;
    if (argc >= 2) N = std::atoi(argv[1]);
    std::vector<float> A(N*N, 1.0f);
    std::vector<float> B(N*N, 1.0f);
    std::vector<float> C(N*N, 0.0f);

  
    auto t0 = std::chrono::high_resolution_clock::now();

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        N, N, N,
        1.0f, A.data(), N,
              B.data(), N,
        0.0f, C.data(), N
    );

  
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  
    double total = 0.0;
    for(auto x : C) total += x;

    std::cout << elapsed_ms << std::endl;

    return 0;
}
