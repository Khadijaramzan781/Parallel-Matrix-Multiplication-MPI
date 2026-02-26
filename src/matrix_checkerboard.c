#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define N 1000   // matrix size (change later for experiments)

int main(int argc, char *argv[])
{
    int rank, size;
    int q;
    double start, end;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    q = (int)sqrt(size);

    // check perfect square
    if (q * q != size) {
        if (rank == 0) {
            printf("Error: Number of processes must be a perfect square\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        printf("MPI initialized with %d processes (%dx%d grid)\n", size, q, q);
    }

    int block_size = N / q;

    // allocate sub-matrices
    double *A = (double *)malloc(block_size * block_size * sizeof(double));
    double *B = (double *)malloc(block_size * block_size * sizeof(double));
    double *C = (double *)malloc(block_size * block_size * sizeof(double));

    // initialize matrices
    for (int i = 0; i < block_size * block_size; i++) {
        A[i] = 1.0;
        B[i] = 1.0;
        C[i] = 0.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);   // sync before timing
    start = MPI_Wtime();

    // block matrix multiplication
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            for (int k = 0; k < block_size; k++) {
                C[i * block_size + j] +=
                    A[i * block_size + k] * B[k * block_size + j];
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);   // sync after computation
    end = MPI_Wtime();

    if (rank == 0) {
        printf("Execution Time = %f seconds\n", end - start);
    }

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}