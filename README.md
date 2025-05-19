# GPUBidimensionalConvolution
## Compilazione
```bash
./compile <version>
```
## Esecuzione
```bash
./main <number of blocks> <number of imgs> <save data>
```

## Test
```bash
./test <version>
```
esegue lo script di compilazione ed inizia i test

## Ottimizzazioni
### V1 -> V2
```c++
__device__ float gaussianBlur(uint16_t i, uint16_t j, float sigma) {
    float denominator = 2.51 * sigma;

    int16_t it = i - ROWS_FILTER / 2;
    int16_t jt = j - COLUMNS_FILTER / 2;

    float exponent = (it * it + jt * jt) / (2 * sigma * sigma);
    return (1.0 / denominator) * ::__expf(-exponent);
}
```
Utilizzo della funzione __expf di CUDA poichè ottimizzata per architetture altamente parallele.\
Meno precisa di expf ma non influisce
___
```c++
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
        if(blurMap[j] == 0) {
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
```
Assegnazione di pixel ai thread invece che righe intere.\
Conversione di tutti i double in float per evitare utilizzazioni di unità a 64 bit (me lo ha chiesto in ginocchio NSight)

### V2 -> V3
```c++
void precomputeFilters(float** cudaPointer) {
    float* filters = (float*)malloc(ROWS_FILTER * COLUMNS_FILTER * sizeof(float) * 255);

    for (uint16_t i = 0; i < 255; i++) {
        computeFilter(filters + i * (ROWS_FILTER * COLUMNS_FILTER), i);
    }

    cudaError_t err = cudaMalloc(cudaPointer, ROWS_FILTER * COLUMNS_FILTER * sizeof(float) * 255);
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaMemcpy(*cudaPointer, filters, ROWS_FILTER * COLUMNS_FILTER * sizeof(float) * 255, cudaMemcpyHostToDevice);
}
```
precalcolo di tutti i filtri possibili
___

```c++
__global__ void bidimensionalConvolution(uint8_t* imgs, uint8_t* blurMap, uint8_t* results, uint16_t nBlocks, uint16_t layersNum, float* filters) {
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
        uint8_t blurValue = blurMap[j % (ROWS_MATRIX * COLUMNS_MATRIX)];
        if(blurValue == 0) {
            results[j] = imgs[j];
            continue;
        }

        uint64_t layer = j / ((uint64_t)ROWS_MATRIX * COLUMNS_MATRIX);
        uint64_t rem   = j % ((uint64_t)ROWS_MATRIX * COLUMNS_MATRIX);
        uint64_t row   = rem / COLUMNS_MATRIX;
        uint64_t col   = rem % COLUMNS_MATRIX;

        results[j] = applyFilter(imgs + layer * ROWS_MATRIX * COLUMNS_MATRIX, row, col, filters + (blurValue - 1) * ROWS_FILTER * COLUMNS_FILTER);
    }
}
```
memorizzazione del valore della blurmap per evitare doppi accessi

### V3 -> V4
spostamento dei filtri precalcolati nella memoria costante ed aumento dei thread per blocco senza aumentare i thread totali.\
Si beneficia dalla constant memory quando più threads di un blocco leggono locazioni vicine (o le stesse) (broadcast hardware).\
Se mettiamo i filtri in constant memory ed aumentiamo il numero di thread per blocco, la probabilità che i threads di
un blocco leggano valori vicini in filters aumenta (in quanto il lavoro assegnato a livello di blocco aumenta).\
Il cerchio sfumato della blurMap contiene valori vicini correlati tra di loro (cambiano poco), quindi i threads leggeranno filtri vicini tra di loro in filters