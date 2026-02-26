#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 500   // Matrix size (change to 720 later)

void init_matrix(double* M, int size) {
    for (int i = 0; i < size * size; i++)
        M[i] = 1.0;
}

int main(int argc, char* argv[]) {

    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int q = (int)sqrt(size);
    if (q * q != size) {
        if (rank == 0)
            printf("Number of processes must be a perfect square\n");
        MPI_Finalize();
        return 0;
    }

    int block = N / q;

    double* A = (double*)malloc(block * block * sizeof(double));
    double* B = (double*)malloc(block * block * sizeof(double));
    double* C = (double*)calloc(block * block, sizeof(double));

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        MPI_Finalize();
        return 0;
    }

    init_matrix(A, block);
    init_matrix(B, block);

    int row = rank / q;
    int col = rank % q;

    MPI_Comm grid;
    int dims[2] = { q, q };
    int periods[2] = { 1, 1 };
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid);

    int left, right, up, down;
    MPI_Cart_shift(grid, 1, -1, &right, &left);
    MPI_Cart_shift(grid, 0, -1, &down, &up);

    MPI_Status status;

    // Initial skewing
    for (int i = 0; i < row; i++)
        MPI_Sendrecv_replace(A, block * block, MPI_DOUBLE, left, 0, right, 0, grid, &status);

    for (int i = 0; i < col; i++)
        MPI_Sendrecv_replace(B, block * block, MPI_DOUBLE, up, 0, down, 0, grid, &status);

    double start = MPI_Wtime();

    for (int step = 0; step < q; step++) {

        for (int i = 0; i < block; i++)
            for (int j = 0; j < block; j++)
                for (int k = 0; k < block; k++)
                    C[i * block + j] += A[i * block + k] * B[k * block + j];

        MPI_Sendrecv_replace(A, block * block, MPI_DOUBLE, left, 0, right, 0, grid, &status);
        MPI_Sendrecv_replace(B, block * block, MPI_DOUBLE, up, 0, down, 0, grid, &status);
    }

    double end = MPI_Wtime();

    if (rank == 0)
        printf("Cannon Algorithm Execution Time: %f seconds\n", end - start);

    free(A);
    free(B);
    free(C);

    MPI_Finalize();
    return 0;
}