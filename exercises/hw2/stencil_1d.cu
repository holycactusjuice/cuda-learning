#include <stdio.h>

#include <algorithm>

#define N 4096
#define RADIUS 3
#define BLOCK_SIZE 16

__global__ void stencil_1d(int *in, int *out) {
    // Create shared memory
    __shared__ int temp[N + 2 * RADIUS];

    // Calculate global and local indices
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // Load from global memory to shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];          // Lower halo
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];  // Upper halo
    }

    __syncthreads();

    // Calculate result
    int result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        result += temp[lindex + offset];
    }

    // Store result in output
    out[gindex] = result;
}

int main() {
    int *in, *out;
    int *d_in, *d_out;
    int size = (N + 2 * RADIUS) * sizeof(int);

    in = (int *)malloc(size);
    out = (int *)malloc(size);

    std::fill_n(in, N + 2 * RADIUS, 1);
    std::fill_n(out, N + 2 * RADIUS, 1);

    // Allocate memory on device
    cudaMalloc((void **)&d_in, size);
    cudaMalloc((void **)&d_out, size);

    // Copy data to device
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, out, size, cudaMemcpyHostToDevice);

    // Launch stencil kernel
    // We need the RADIUS offset since the first RADIUS elements are halo
    // elements
    stencil_1d<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_in + RADIUS, d_out + RADIUS);

    // Copy results back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);

    // Error checking
    for (int i = 0; i < N + 2 * RADIUS; i++) {
        // Halo elements should be 1
        if (i < RADIUS || i >= N + RADIUS) {
            if (out[i] != 1) {
                printf("Mismatch at index %d, should be: %d, is: %d\n", i, 1,
                       out[i]);
            }
        } else {
            if (out[i] != 1 + 2 * RADIUS) {
                printf("Mismatch at index %d, should be: %d, is: %d\n", i,
                       1 + 2 * RADIUS, out[i]);
            }
        }
    }

    free(in);
    free(out);
    cudaFree(d_in);
    cudaFree(d_out);

    printf("Done!\n");

    return 0;
}