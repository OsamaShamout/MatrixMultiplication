%%cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define BLOCK_SIZE 16

__global__ void MatrixMult(float* M, float* N, float* P, int height, int width, int depth)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < depth) {
        float pvalue = 0;
        for (int k = 0; k < width; k++) {
            pvalue += M[row * width + k] * N[k * depth + col];
        }
        P[row * depth + col] = pvalue;
    }
}

int main() {
    int M = 1500;
    int N = 1500;
    int K = 1500;

    //Allocate mem for host
    float *h_M = (float*)malloc(sizeof(float) * M * N);
    float *h_N = (float*)malloc(sizeof(float) * M * K);
    float *h_P = (float*)malloc(sizeof(float) * N * K);

    srand(time(NULL));
    for (int i = 0; i < N * M; i++) {
        h_M[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < M * K; i++) {
        h_N[i] = rand() / (float)RAND_MAX;
    }

    //Allocate mem for device
    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_N, sizeof(float) * M * K);
    cudaMalloc((void**)&d_M, sizeof(float)* M * N);
    cudaMalloc((void**)&d_P, sizeof(float) * N * K);
    cudaMemcpy(d_M, h_M, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, sizeof(float) * M * K, cudaMemcpyHostToDevice);
        
    //Set up config for parallelization
    dim3 dimGrid(ceil(K / (float)BLOCK_SIZE), ceil(N / (float)BLOCK_SIZE), 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //Invoke Kernel
    MatrixMult<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, N, M, K);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_P, d_P, sizeof(float)*N*K, cudaMemcpyDeviceToHost);

    printf("Kernel execution time: %f ms\n", elapsed_time);

    //Host mem free
    free(h_M);
    free(h_N);
    free(h_P);

    //Device mem free
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
