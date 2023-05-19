%%cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define TITLE_SIZE 16

// Kernel function with tiling
__global__ void MatrixMultTiled(float* M, float* N, float* P, int width, int height, int depth)
{
    __shared__ float Ms[TITLE_SIZE][TITLE_SIZE];
    __shared__ float Ns[TITLE_SIZE][TITLE_SIZE];

    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x; 
    int ty = threadIdx.y;

    // Identify the row and column of the P element to work on
    int Row = by * TITLE_SIZE + ty;
    int Col = bx * TITLE_SIZE + tx;

    float Pvalue = 0;

    // Loop over the M and N tiles required to compute the P element
    for (int m = 0; m < (width-1)/TITLE_SIZE+1; ++m) {

        // Collaborative loading of M and N tiles into shared memory
        if (Row < width && m*TITLE_SIZE+tx < height) 
            Ms[ty][tx] = M[Row*width + (m*TITLE_SIZE + tx)];
        else
            Ms[ty][tx] = 0.0;

        if (m*TITLE_SIZE+ty < width && Col < depth)
            Ns[ty][tx] = N[(m*TITLE_SIZE + ty)*depth + Col];
        else
            Ns[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TITLE_SIZE; ++k) {
            Pvalue += Ms[ty][k] * Ns[k][tx];
        }
        __syncthreads();
    }
    if (Row < width && Col < depth)
        P[Row*depth + Col] = Pvalue;
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
    dim3 dimGrid(ceil(K / (float)TITLE_SIZE), ceil(N / (float)TITLE_SIZE), 1);
    dim3 dimBlock(TITLE_SIZE, TITLE_SIZE, 1);

    float elapsed_time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    //Invoke Kernel
    MatrixMultTiled<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, N, M, K);

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

function TiledMatrixMultiplication(M, N, P, width, height, depth)
Define shared matrices Ms and Ns with TITLE_SIZE
    Calculate Row and Col
        Initialize Pvalue to 0 Loop over tiles
            Load elements from M and N into shared memory Ms and Ns
                Synchronize threads
                    Loop over TITLE_SIZE
                        Multiply corresponding elements of Ms and Ns and add to Pvalue
                            Synchronize threads
                                Write Pvalue to P[Row * depth + Col]
