#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg)                                        \
    do {                                                            \
        cudaError_t __err = cudaGetLastError();                     \
        if (__err != cudaSuccess) {                                 \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", msg, \
                    cudaGetErrorString(__err), __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n");             \
            exit(1);                                                \
        }                                                           \
    } while (0)

const int DSIZE = 4096;
const int block_size = 256;

__global__ void vadd(const float *A, const float *B, float *C, int ds) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ds) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // instantiate host and device arrays
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    h_C = new float[DSIZE];

    // fill host arrays with random values
    // fill result array with zeroes
    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_C[i] = 0;
    }

    // calculate size needed, in bytes
    int size = DSIZE * sizeof(float);

    // allocate memory on device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaCheckErrors("cudaMalloc error");

    // copy arrays from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy host to device error");

    // calculate number of blocks needed, rounding up
    int n_blocks = (DSIZE + block_size - 1) / block_size;

    // kernel call
    vadd<<<n_blocks, block_size>>>(d_A, d_B, d_C, DSIZE);

    // copy results from device back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaCheckErrors(
        "cudaMemcpy device to host error or kernel execution failure");

    // print out first index
    printf("A[0] = %f\n", h_A[0]);
    printf("B[0] = %f\n", h_B[0]);
    printf("C[0] = %f\n", h_C[0]);

    return 0;
}