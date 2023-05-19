#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 500

void MatrixMultSeq(float M[SIZE][SIZE], float N[SIZE][SIZE], float P[SIZE][SIZE])
{
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            P[i][j] = 0.0;
            for (int k = 0; k < SIZE; k++) {
                P[i][j] += M[i][k] * N[k][j];
            }
        }
    }
}

int main() {
    float M[SIZE][SIZE];
    float N[SIZE][SIZE];
    float P[SIZE][SIZE];

    srand(time(NULL));
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            M[i][j] = rand() / (float)RAND_MAX;
            N[i][j] = rand() / (float)RAND_MAX;
        }
    }

    clock_t start, end;
    double cpu_time;

    start = clock();

    MatrixMultSeq(M, N, P);

    end = clock();

    cpu_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Execution time: %f s\n", cpu_time);

    return 0;
}
