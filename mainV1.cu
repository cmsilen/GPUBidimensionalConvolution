#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <locale.h>
#include <cuda.h>

#define N 512  // Dimensione dei vettori

#define SIGMA_MAX 0.5
#define ROWS_MATRIX 2160
#define COLUMNS_MATRIX 1440
#define MAX_NUMBER 255
#define MIN_NUMBER 0

void imgsCudaMalloc(uint8_t n, uint16_t rows, uint16_t cols, uint8_t** cudaPointer);

// Kernel CUDA
__global__ void vectorAdd(float *A, float *B, float *C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


double example() {
    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);

    // Allocazione host
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(N * sizeof(float));
    h_B = (float*)malloc(N * sizeof(float));
    h_C = (float*)malloc(N * sizeof(float));

    // Inizializzazione
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocazione device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copia dai dati da host a device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;


    QueryPerformanceCounter(&start);

    // Lancio del kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C);

    QueryPerformanceCounter(&end);


    double elapsedTime = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;

    // Copia del risultato da device a host
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
    // Verifica e stampa
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Liberazione memoria
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return elapsedTime;
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("./main <N_THREADS> <N_IMGS> <ROWS_FILTER> <saveData>\n");
        return 1; // Esce con codice di errore
    }
    // Converte gli argomenti in interi
    int NThread = atoi(argv[1]);
    int NImgs = atoi(argv[2]);
    int ROWS_FILTER = atoi(argv[3]);
    int COLUMNS_FILTER = ROWS_FILTER;
    int LAYERS_NUM = NImgs * 3;
    int saveData = atoi(argv[4]);

    uint8_t* d_imgs = nullptr;        //d_imgs[n_img][i][j] = d_imgs[n_img * (rows * cols) + i * cols + j];
    imgsCudaMalloc(LAYERS_NUM, ROWS_MATRIX, COLUMNS_MATRIX, &d_imgs);
    printf("ok");
    cudaFree(d_imgs);

    //double elapsedTime = experiment();
    //printf("Elapsed time = %f ms\n", elapsedTime);

    return 0;
}

int16_t g_seed = 10;
int16_t randomNumber(int16_t min, int16_t max) {
    g_seed = (214013*g_seed+2531011);
    return ((g_seed>>16)&0x7FFF) % (max - min + 1) + min;

    //return rand() % (max - min + 1) + min;
}

void fillMatrix(uint8_t* matrix, uint16_t rows, uint16_t cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = randomNumber(MIN_NUMBER, MAX_NUMBER);
        }
    }
}

void imgsCudaMalloc(uint8_t n, uint16_t rows, uint16_t cols, uint8_t** cudaPointer) {
    uint8_t* img = (uint8_t*)malloc(cols * rows * sizeof(uint8_t));
    int totalSize = n * cols * rows;

    cudaMalloc(cudaPointer, totalSize * sizeof(uint8_t));

    for (int i = 0; i < n; i++) {
        fillMatrix(img, rows, cols);
        cudaMemcpy(cudaPointer + i * (rows * cols), img, rows * cols * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    free(img);
}