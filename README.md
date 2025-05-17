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
__device__ double gaussianBlur(uint16_t i, uint16_t j, double sigma) {
    double denominator = 2.51 * sigma;

    uint16_t it = i - ROWS_FILTER / 2;
    uint16_t jt = j - COLUMNS_FILTER / 2;

    double exponent = -(it * it + jt * jt) / (2 * sigma * sigma);
    return (1.0 / denominator) * ::exp(exponent);
}
```
Utilizzo della funzione exp di CUDA poichè ottimizzata per architetture altamente parallele
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
