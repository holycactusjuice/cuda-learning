#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <chrono>

// #define N 1024

__global__ void gpuAdd(int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

void cpuAdd(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void randomize(int *a, int n) {
    srand (time(NULL));
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 1000;
    }
}

int main() {
    int N = INT_MAX / sizeof(int);

    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    int size = N * sizeof(int);

    cudaMalloc((void **) &d_a, size);
    cudaMalloc((void **) &d_b, size);
    cudaMalloc((void **) &d_c, size);

    a = (int *) malloc(size); randomize(a, N);
    b = (int *) malloc(size); randomize(b, N);
    c = (int *) malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    auto time_start_gpu = std::chrono::high_resolution_clock::now();

    gpuAdd<<<1, N>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    auto time_end_gpu = std::chrono::high_resolution_clock::now();

    std::cout << "Time taken by GPU: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end_gpu - time_start_gpu).count() << " microseconds" << std::endl;

    auto time_start_cpu = std::chrono::high_resolution_clock::now();

    cpuAdd(a, b, c, N);

    auto time_end_cpu = std::chrono::high_resolution_clock::now();

    std::cout << "Time taken by CPU: " << std::chrono::duration_cast<std::chrono::microseconds>(time_end_cpu - time_start_cpu).count() << " microseconds" << std::endl;

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}