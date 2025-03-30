#include <stdio.h>
#include <time.h>

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
const int block_size = 16;
const float A_val = 1.0f;
const float B_val = 2.0f;

__global__ void matmul(const float *A, const float *B, float *C, int ds) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int t = idx + idy * ds;

    if ((idx < ds) && (idy < ds)) {
        float temp = 0;
        // C[t] = (idy'th row of A) dot (idx'th col of B)
        // for A, i moves across the idy'th row
        // for B, i moves down the idx'th col
        for (int i = 0; i < ds; i++) {
            temp += A[idy * ds + i] * B[ds * i + idx];
        }
        C[t] = temp;
    }
}

int main() {
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    h_A = new float[DSIZE * DSIZE];
    h_B = new float[DSIZE * DSIZE];
    h_C = new float[DSIZE * DSIZE];

    for (int i = 0; i < DSIZE * DSIZE; i++) {
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
    }

    int size = DSIZE * DSIZE * sizeof(float);

    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaCheckErrors("cudaMalloc error");

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    cudaCheckErrors("cudaMemcpy H2D error");

    // define dimensions of block and grid

    // dim3 variable holds up to 3 dimensions
    // we have already defined block_size above
    dim3 block(block_size, block_size);
    // each grid dim is responsible for DSIZE threads
    dim3 grid((DSIZE + block.x - 1) / block.x, (DSIZE + block.y - 1) / block.y);

    // launch kernel
    matmul<<<grid, block>>>(d_A, d_B, d_C, DSIZE);

    // copy results from device back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaCheckErrors("cudaMemcpy D2H error");

    for (int i = 0; i < DSIZE * DSIZE; i++) {
        if (h_C[i] != A_val * B_val * DSIZE) {
            printf("Error at index %d\n", i);
        }
    }
    printf("Complete\n");
    return 0;
}