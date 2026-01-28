#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#define PI 3.14159265358979323846

int main(int argc, char** argv) {
    int N = 4096;
    if (argc >= 2) N = std::atoi(argv[1]);

    std::vector<float> input(N, 1.0f);
    std::vector<float> out_real(N, 0.0f);
    std::vector<float> out_imag(N, 0.0f);

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < N; ++k) {
        float real = 0.0f, imag = 0.0f;
        for (int n = 0; n < N; ++n) {
            float angle = -2.0f * PI * k * n / N;
            real += input[n] * std::cos(angle);
            imag += input[n] * std::sin(angle);
        }
        out_real[k] = real;
        out_imag[k] = imag;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double mag_sum = 0.0;
    for (int i = 0; i < N; i++)
        mag_sum += std::sqrt(out_real[i]*out_real[i] + out_imag[i]*out_imag[i]);

    std::cout << ms << std::endl;
    return 0;
}
