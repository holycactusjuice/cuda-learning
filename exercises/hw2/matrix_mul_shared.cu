#include <stdio.h>

#include <time.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

const int N = 8192;
const int BLOCK_SIZE = 32;
const float A_val = 3.0f;
const float B_val = 2.0f;

__global__ void mmul(float *A, float *B, float *C, int n) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    // Exit if out of bounds
    if (idx >= n || idy >= n) return;

    // Accumulating value which will be stored in C[idy][idx] at the end
    float temp = 0.0f;

    for (int i = 0; i < N/BLOCK_SIZE; i++) {
        // Load global memory into shared memory
        As[threadIdx.y][threadIdx.x] = A[idy * N + threadIdx.x + i * BLOCK_SIZE]; // Row offset is idy * N, col offset is i * BLOCK_SIZE + threadIdx.x
        Bs[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * N + idx]; // Row offset is (i * BLOCK_SIZE + threadIdx.y) * N, col offset is idx
    
        // Sync to make sure all values have been loaded
        __syncthreads();

        // Accumulate partial dot product in temp
        for (int j = 0; j < BLOCK_SIZE; j++) {
            temp += As[threadIdx.y][j] * Bs[j][threadIdx.x];
        }

        __syncthreads();
    }

    C[idy * N + idx] = temp;
}

int main() {
    clock_t t0, t1, t2;

    t0 = clock();

    float *A, *B, *C, *d_A, *d_B, *d_C;
    A = new float[N*N];
    B = new float[N*N];
    C = new float[N*N];
    
    for (int i = 0; i < N*N; i++) {
        A[i] = A_val;
        B[i] = B_val;
        C[i] = 0;
    }

    t1 = clock();
    double tInit = ((double)(t1 - t0)) / CLOCKS_PER_SEC;
    printf("Init took %f seconds. Begin compute\n", tInit);

    int size = N*N*sizeof(float);

    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);
    cudaCheckErrors("cudaMalloc error");

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D error");

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N+block.x-1)/block.x, (N+block.y-1)/block.y);
    mmul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaCheckErrors("kernel launch error");

    // Copy results back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H error");

    t2= clock();
    double tCompute = ((double)(t2 - t1)) / CLOCKS_PER_SEC;
    printf("Compute took %f seconds\n", tCompute);

    for (int i = 0; i < N*N; i++) {
        if (C[i] != A_val * B_val * N) {
            printf("Mismatch at %d; expected: %f, got: %f\n", i, A_val * B_val * N, C[i]);
        }
    }

    printf("Done!\n");
    return 0;
}