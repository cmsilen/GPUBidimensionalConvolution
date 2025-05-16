#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>
#include <locale.h>
#include <cuda.h>

#define SIGMA_MAX 0.5
#define ROWS_MATRIX 2160
#define COLUMNS_MATRIX 1440
#define ROWS_FILTER 7
#define COLUMNS_FILTER ROWS_FILTER
#define MAX_NUMBER 255
#define MIN_NUMBER 0

// depends on sigma and the coords of the filter
__device__ double gaussianBlur(uint16_t i, uint16_t j, double sigma) {
    double denominator = sqrt(2 * 3.14 * sigma * sigma);
    double exponent = -(i * i + j * j) / (2 * sigma * sigma);
    return (1.0 / denominator) * exp(exponent);
}

// depends on the coords of the matrix
__device__ double sigmaFunction(uint16_t i, uint16_t j, uint8_t* blurMap) {
    return blurMap[i * COLUMNS_MATRIX + j] * SIGMA_MAX;
}

// to compute the filter given the coords of the matrix
__device__ void computeFilter(double* filter, uint16_t row, uint16_t col, uint8_t* blurMap) {
    for (uint16_t i = 0; i < ROWS_FILTER; i++) {
        for (uint16_t j = 0; j < COLUMNS_FILTER; j++) {
            filter[i * COLUMNS_FILTER + j] = gaussianBlur(i, j, sigmaFunction(row, col, blurMap));
        }
    }
}

__device__ uint8_t applyFilter(uint8_t* matrix, uint16_t x, uint16_t y, double* filter) {
    double result = 0;
    uint16_t i, j;

    for (i = 0; i < ROWS_FILTER; i++) {
        for (j = 0; j < COLUMNS_FILTER; j++) {
            if (x - (ROWS_FILTER / 2) + i < 0 || x - (ROWS_FILTER / 2) + i >= ROWS_MATRIX ||
                y - (COLUMNS_FILTER / 2) + j < 0 || y - (COLUMNS_FILTER / 2) + j >= COLUMNS_MATRIX)
                continue;
            result += matrix[(x - (ROWS_FILTER / 2) + i) * COLUMNS_FILTER + (y - (COLUMNS_FILTER / 2) + j)] * filter[i * COLUMNS_FILTER + j];
        }
    }

    if (result > 255)
        return 255;
    return result;
}

__global__ void bidimensionalConvolution(uint8_t* imgs, uint8_t* blurMap, uint8_t* results, uint16_t nThreads, uint16_t layersNum) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= ROWS_MATRIX)
        return;

    double filter[ROWS_FILTER * COLUMNS_FILTER];

    uint16_t rowsPerThread = ROWS_MATRIX / (nThreads * blockDim.x);
    uint16_t start = idx * rowsPerThread;
    uint16_t end = (idx + 1) * rowsPerThread;

    for(uint16_t i = 0; i < layersNum; i++) {
        for(uint16_t j = 0; j < ROWS_MATRIX; j++) {
            for(uint16_t k = start; k < end; k++) {
                computeFilter(filter, j, k, blurMap);
                results[i * (ROWS_MATRIX * COLUMNS_MATRIX) + j * COLUMNS_MATRIX + k] = applyFilter(imgs + i * (ROWS_MATRIX * COLUMNS_MATRIX), j, k, filter);
            }
        }
    }
}

void imgsCudaMalloc(uint8_t n, uint16_t rows, uint16_t cols, uint8_t** cudaPointer, uint8_t toFill);
void blurMapCudaMalloc(uint16_t rows, uint16_t cols, uint8_t** cudaPointer);

uint16_t LAYERS_NUM;
uint8_t* imgs;
uint8_t* blurMap;

double experiment(uint16_t nThreads) {
    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);

    // allocazione
    uint8_t* d_imgs = nullptr;        //d_imgs[n_img][i][j] = d_imgs[n_img * (rows * cols) + i * cols + j];
    uint8_t* d_blurMap = nullptr;
    uint8_t* d_results = nullptr;
    imgsCudaMalloc(LAYERS_NUM, ROWS_MATRIX, COLUMNS_MATRIX, &d_imgs, 1);
    blurMapCudaMalloc(ROWS_MATRIX, COLUMNS_FILTER, &d_blurMap);
    imgsCudaMalloc(LAYERS_NUM, ROWS_MATRIX, COLUMNS_MATRIX, &d_results, 0);

    int threadsPerBlock = nThreads;
    int blocksPerGrid = (ROWS_MATRIX + threadsPerBlock - 1) / threadsPerBlock;

    QueryPerformanceCounter(&start);

    // Lancio del kernel
    bidimensionalConvolution<<<blocksPerGrid, threadsPerBlock>>>(d_imgs, d_blurMap, d_results, nThreads, LAYERS_NUM);

    //controllo errori di lancio
    cudaError_t err = cudaGetLastError();  // controlla errori di lancio kernel
    if (err != cudaSuccess) {
        printf("Errore lancio kernel: %s\n", cudaGetErrorString(err));
    }
    cudaError_t errSync = cudaDeviceSynchronize();
    QueryPerformanceCounter(&end);
    //controllo errori finali
    if (errSync != cudaSuccess) {
        printf("Errore runtime kernel: %s\n", cudaGetErrorString(errSync));
    }

    double elapsedTime = (double)(end.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0;

    cudaFree(d_imgs);
    cudaFree(d_blurMap);
    cudaFree(d_results);
    return elapsedTime;
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("./main <N_THREADS> <N_IMGS> <saveData>\n");
        return 1; // Esce con codice di errore
    }
    // Converte gli argomenti in interi
    int NThread = atoi(argv[1]);
    int NImgs = atoi(argv[2]);
    LAYERS_NUM = NImgs * 3;
    int saveData = atoi(argv[3]);

    double elapsedTime = experiment(NThread);
    printf("Elapsed time = %f ms\n", elapsedTime);

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

void imgsCudaMalloc(uint8_t n, uint16_t rows, uint16_t cols, uint8_t** cudaPointer, uint8_t toFill) {
    uint8_t* img = (uint8_t*)malloc(cols * rows * sizeof(uint8_t));
    int totalSize = n * cols * rows;

    cudaError_t err = cudaMalloc(cudaPointer, totalSize * sizeof(uint8_t));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        if(toFill > 0) {
            fillMatrix(img, rows, cols);
        }
        cudaMemcpy((*cudaPointer) + i * (rows * cols), img, rows * cols * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    free(img);
}

void disegna_cerchio_sfumato(uint8_t* matrice, int width, int height) {
    int centerX = width / 2;
    int centerY = height / 2;
    float radius = width / 3.0f;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int dx = x - centerX;
            int dy = y - centerY;
            float distanza = sqrtf(dx * dx + dy * dy);

            if (distanza <= radius) {
                float valore = 255.0f * (1.0f - (distanza / radius));
                matrice[x * width + y] = (uint8_t)(valore + 0.5f); // arrotondamento
            } else {
                matrice[x * width + y] = 0;
            }
        }
    }
}

void blurMapCudaMalloc(uint16_t rows, uint16_t cols, uint8_t** cudaPointer) {
    uint8_t* img = (uint8_t*)malloc(cols * rows * sizeof(uint8_t));
    disegna_cerchio_sfumato(img, cols, rows);

    int totalSize = cols * rows;
    cudaError_t err = cudaMalloc(cudaPointer, totalSize * sizeof(uint8_t));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaMemcpy(*cudaPointer, img, totalSize * sizeof(uint8_t), cudaMemcpyHostToDevice);
    free(img);
}