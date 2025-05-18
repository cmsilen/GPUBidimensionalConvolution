#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <windows.h>
#include <locale.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define SIGMA_MAX 0.5
#define ROWS_MATRIX 512
#define COLUMNS_MATRIX 512
#define ROWS_FILTER 7
#define COLUMNS_FILTER ROWS_FILTER
#define MAX_NUMBER 255
#define MIN_NUMBER 0
#define THREADS_PER_BLOCK 32
#define DEBUG 0

int read_pgm(const char* filename, uint8_t* img, int* width, int* height);
int write_pgm(const char* filename, uint8_t* out, int width, int height);
void concatStringNumber(char *str, int numero);

// depends on sigma and the coords of the filter
__device__ float gaussianBlur(uint16_t i, uint16_t j, float sigma) {
    float denominator = 2.51 * sigma;

    int16_t it = i - ROWS_FILTER / 2;
    int16_t jt = j - COLUMNS_FILTER / 2;

    float exponent = (it * it + jt * jt) / (2 * sigma * sigma);
    return (1.0 / denominator) * ::__expf(-exponent);
}

// depends on the coords of the matrix
__device__ float sigmaFunction(uint16_t i, uint16_t j, uint8_t* blurMap) {
    return blurMap[i * COLUMNS_MATRIX + j] * SIGMA_MAX;
}

// to compute the filter given the coords of the matrix
__device__ void computeFilter(float* filter, uint16_t row, uint16_t col, uint8_t* blurMap) {
    for (uint16_t i = 0; i < ROWS_FILTER; i++) {
        for (uint16_t j = 0; j < COLUMNS_FILTER; j++) {
            filter[i * COLUMNS_FILTER + j] = gaussianBlur(i, j, sigmaFunction(row, col, blurMap));
        }
    }
}

__device__ uint8_t applyFilter(uint8_t* matrix, uint16_t x, uint16_t y, float* filter) {
    float result = 0;
    uint16_t i, j;

    uint16_t startX = 0;
    uint16_t startY = 0;
    uint16_t HALF_ROW = (ROWS_FILTER / 2);
    uint16_t HALF_COLUMN = (COLUMNS_FILTER / 2);
    if(x < HALF_ROW) startX = HALF_ROW - x;
    if(y < HALF_COLUMN) startY = HALF_COLUMN - y;

    uint16_t endX = ROWS_FILTER;
    uint16_t endY = COLUMNS_FILTER;
    if(x >= ROWS_MATRIX - HALF_ROW) endX = HALF_ROW + ROWS_MATRIX - x;
    if(y >= COLUMNS_MATRIX - HALF_COLUMN) endY = HALF_COLUMN + COLUMNS_MATRIX - y;

    uint16_t k = x - HALF_ROW + startX;
    for (i = startX; i < endX; i++) {
        uint16_t h = y - HALF_COLUMN + startY;
        for (j = startY; j < endY; j++) {
            result += matrix[k * COLUMNS_MATRIX + h] * filter[i * COLUMNS_FILTER + j];
            h++;
        }
        k++;
    }

    if (result > 255)
        return 255;
    return result;
}

/*
    SCHEDULAZIONE:
    la gpu è composta da Streaming Multiprocessor (SM) che eseguono insiemi di 32 threads (warps).
    Al lancio del kernel, uno SM riceve blocksPerGrid blocchi definiti in bidimensionalConvolution<<<blocksPerGrid, threadsPerBlock>>>.
    Un blocco rimane sullo SM fino alla fine dell'elaborazione.
    Il blocco viene diviso in warps da 32 threads. I warps vengono schedulati per l'esecuzione sullo SM.
    Lo SM può arrivare in genere a 64 warps attivi.
    In base al numero di threads che assegnamo per blocco, andiamo a definire il numero di warps usati per blocco.
    Se il numero di threads per blocco non è un multiplo di 32, per ogni blocco si creerà un warp parziale (con meno di 32 threads) che andrà
    a peggiorare l'utilizzazione dell'architettura.
    Pochi warps -> bassa utilizzazione
    Troppi warps -> inefficienza
*/

__global__ void bidimensionalConvolution(uint8_t* imgs, uint8_t* blurMap, uint8_t* results, uint16_t nBlocks, uint16_t layersNum) {
    float filter[ROWS_FILTER * COLUMNS_FILTER];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int totThreads = nBlocks * THREADS_PER_BLOCK;

    if (idx >= totThreads)
        return;

    uint64_t basePixels = (ROWS_MATRIX * COLUMNS_MATRIX * layersNum) / totThreads;
    uint64_t extraPixels = (ROWS_MATRIX * COLUMNS_MATRIX * layersNum) % totThreads;

    uint64_t start, end;

    if (idx < extraPixels) {
        start = idx * (basePixels + 1);
        end = start + basePixels + 1;
    } else {
        start = idx * basePixels + extraPixels;
        end = start + basePixels;
    }

    for(uint64_t j = start; j < end; j++) {
        if(blurMap[j % (ROWS_MATRIX * COLUMNS_MATRIX)] == 0) {
            results[j] = imgs[j];
            continue;
        }

        uint64_t layer = j / ((uint64_t)ROWS_MATRIX * COLUMNS_MATRIX);
        uint64_t rem   = j % ((uint64_t)ROWS_MATRIX * COLUMNS_MATRIX);
        uint64_t row   = rem / COLUMNS_MATRIX;
        uint64_t col   = rem % COLUMNS_MATRIX;

        computeFilter(filter, row, col, blurMap);
        results[j] = applyFilter(imgs + layer * ROWS_MATRIX * COLUMNS_MATRIX, row, col, filter);
    }
}

void imgsCudaMalloc(uint8_t n, uint16_t rows, uint16_t cols, uint8_t** cudaPointer);
void outCudaMalloc(uint8_t n, uint16_t rows, uint16_t cols, uint8_t** cudaPointer);
void blurMapCudaMalloc(uint16_t rows, uint16_t cols, uint8_t** cudaPointer);

uint16_t LAYERS_NUM;
uint8_t* imgs;
uint8_t* blurMap;

double experiment(uint16_t nBlocks) {
    LARGE_INTEGER start, end, freq;
    QueryPerformanceFrequency(&freq);

    // allocazione
    uint8_t* d_imgs = nullptr;
    uint8_t* d_blurMap = nullptr;
    uint8_t* d_results = nullptr;
    imgsCudaMalloc(LAYERS_NUM, ROWS_MATRIX, COLUMNS_MATRIX, &d_imgs);
    blurMapCudaMalloc(ROWS_MATRIX, COLUMNS_MATRIX, &d_blurMap);
    outCudaMalloc(LAYERS_NUM, ROWS_MATRIX, COLUMNS_MATRIX, &d_results);

    if(DEBUG) {
        printf("starting %d threads\n", nBlocks * THREADS_PER_BLOCK);
        printf("rows per thread: %d\n", ROWS_MATRIX / (nBlocks * THREADS_PER_BLOCK));
    }

    QueryPerformanceCounter(&start);

    // Lancio del kernel
    bidimensionalConvolution<<<nBlocks, THREADS_PER_BLOCK>>>(d_imgs, d_blurMap, d_results, nBlocks, LAYERS_NUM);

    //controllo errori di lancio
    cudaError_t err = cudaGetLastError();  // controlla errori di lancio kernel
    if (err != cudaSuccess) {
        printf("Errore lancio kernel: %s\n", cudaGetErrorString(err));
    }
    cudaError_t errSync = cudaDeviceSynchronize();
    QueryPerformanceCounter(&end);

    // ---------------- SALVO IMMAGINI ---------------- //
    uint8_t* host_results = (uint8_t*)malloc(ROWS_MATRIX * COLUMNS_MATRIX);
    for (int g = 0; g < LAYERS_NUM; g++) {
        err = cudaMemcpy(
            host_results,
            d_results,
            ROWS_MATRIX * COLUMNS_MATRIX,
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        } else {
            char filename[20] = "out_";
            concatStringNumber(filename, g); strcat(filename, ".pgm\0");
            write_pgm(filename, host_results, 512, 512);
        }
    }
    // ------------------------------------------------ //

    // ----------------- SALVO BLURMAP ---------------- //
    err = cudaMemcpy(
        host_results,
        d_blurMap,
        ROWS_MATRIX * COLUMNS_MATRIX,
        cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("cudaMemcpy failed: %s\n", cudaGetErrorString(err));
    } else
        write_pgm("out_blur.pgm", host_results, 512, 512);
    // ------------------------------------------------ //

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

void concatStringNumber(char *str, int numero) {
    char numStr[20];
    sprintf(numStr, "%d", numero);

    strcat(str, numStr);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf("./main <N_BLOCKS> <N_IMGS> <saveData>\n");
        return 1; // Esce con codice di errore
    }
    // Converte gli argomenti in interi
    uint16_t NBlocks = atoi(argv[1]);
    uint16_t NImgs = atoi(argv[2]);
    LAYERS_NUM = NImgs;
    uint16_t saveData = atoi(argv[3]);
    uint16_t realNBlocks = NBlocks;

    if(NBlocks * THREADS_PER_BLOCK > ROWS_MATRIX * LAYERS_NUM) {
        NBlocks = (ROWS_MATRIX * LAYERS_NUM) / THREADS_PER_BLOCK;
        printf("thread limitati a %d\n", NBlocks);
    }

    double elapsedTime = experiment(NBlocks);
    printf("Elapsed time = %f ms\n", elapsedTime);

    if(!saveData) {
        return 0;
    }

    if(setlocale(LC_NUMERIC, "Italian_Italy.1252") == NULL) {
        printf("Failed to set locale\n");
        return 1;
    }

    char filename[100] = "resultsV1/executionTime_";
    concatStringNumber(filename, NImgs);
    strcat(filename, "IMGS.csv\0");
    FILE* file = fopen(filename, "r");
    int exists = file != NULL;
    if (file != NULL) {
        exists = 1;
        fclose(file);
    }
    file = fopen(filename, "a");

    if(exists == 0) {
        fprintf(file, "Threads;NImgs;RowsFilter;executionTime\n");
    }

    fprintf(file, "%d;%d;%d;%.3f\n", realNBlocks * THREADS_PER_BLOCK, NImgs, ROWS_FILTER, elapsedTime);
    fclose(file);
    return 0;
}

static uint32_t rng_state = 123456789;
uint8_t randomNumber() {
    rng_state = 1664525 * rng_state + 1013904223;
    return (uint8_t)(rng_state >> 24);
}

void fillMatrix(uint8_t* matrix, uint16_t rows, uint16_t cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = randomNumber();
        }
    }
}

void imgsCudaMalloc(uint8_t n, uint16_t rows, uint16_t cols, uint8_t** cudaPointer) {
    uint8_t* img = (uint8_t*)malloc(cols * rows * sizeof(uint8_t));
    int totalSize = n * cols * rows;

    cudaError_t err = cudaMalloc(cudaPointer, totalSize * sizeof(uint8_t));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    int width, height;
    if (read_pgm("input_img.pgm", img, &width, &height) != 0) {
        printf("Errore lettura dell'immagine: %dx%d\n", width, height);
        return;
    }

    for (int i = 0; i < n; i++) {
        cudaMemcpy((*cudaPointer) + i * (rows * cols), img, rows * cols * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }

    free(img);
}

void outCudaMalloc(uint8_t n, uint16_t rows, uint16_t cols, uint8_t** cudaPointer) {
    uint8_t* img = (uint8_t*)malloc(cols * rows * sizeof(uint8_t));
    int totalSize = n * cols * rows;

    cudaError_t err = cudaMalloc(cudaPointer, totalSize * sizeof(uint8_t));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    for (int i = 0; i < n; i++) {
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
                matrice[y * width + x] = (uint8_t)(valore + 0.5f); // arrotondamento
            } else {
                matrice[y * width + x] = 0;
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

int read_pgm(const char* filename, uint8_t* img, int* width, int* height) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Errore nell'apertura del file");
        return -1;
    }

    char line[256];
    
    // Legge il tipo PGM (deve essere "P2")
    if (!fgets(line, sizeof(line), file)) {
        fclose(file);
        return -1;
    }
    if (strncmp(line, "P2", 2) != 0) {
        fprintf(stderr, "Formato non supportato (non P2)\n");
        fclose(file);
        return -1;
    }

    // Salta commenti e legge le dimensioni
    do {
        if (!fgets(line, sizeof(line), file)) {
            fclose(file);
            return -1;
        }
    } while (line[0] == '#');

    // Legge width e height
    if (sscanf(line, "%d %d", width, height) != 2) {
        fprintf(stderr, "Errore nella lettura delle dimensioni\n");
        fclose(file);
        return -1;
    }

    // Legge il valore massimo di grigio (tipicamente 255)
    int maxval;
    if (!fgets(line, sizeof(line), file) || sscanf(line, "%d", &maxval) != 1) {
        fprintf(stderr, "Errore nella lettura del valore massimo\n");
        fclose(file);
        return -1;
    }

    // Legge i pixel e li salva in img
    for (int y = 0; y < *height; y++) {
        for (int x = 0; x < *width; x++) {
            int pixel;
            if (fscanf(file, "%d", &pixel) != 1) {
                fprintf(stderr, "Errore nella lettura dei pixel\n");
                fclose(file);
                return -1;
            }
            img[y * (*width) + x] = (uint8_t)pixel;
        }
    }

    fclose(file);
    return 0;
}

int write_pgm(const char* filename, uint8_t* out, int width, int height) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Errore nell'apertura del file per la scrittura");
        return -1;
    }

    // Scrive l'header del PGM in formato ASCII
    fprintf(file, "P2\n");
    fprintf(file, "# Generated by write_pgm\n");
    fprintf(file, "%d %d\n", width, height);
    fprintf(file, "255\n");  // Massimo valore di grigio

    // Scrive i pixel riga per riga
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int val = out[y * width + x];
            if (val < 0) val = 0;
            if (val > 255) val = 255;
            fprintf(file, "%d ", val);
        }
        fprintf(file, "\n");
    }

    fclose(file);
    return 0;
}
