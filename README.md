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

## TEST TPB
### V1
il test ha mostrato che la configurazione TPB migliore è 32.\
il motivo è che essendo i threads completamente indipendenti, si preferisce la larghezza del blocco minore in modo tale da distribuire meglio i\
warp totali tra tutti gli Streaming Multiprocessor.
### V2
nessuna differenza sostanziale tra configurazioni: leggermente migliore TPB 32.\
impossibile usare TPB 1024 in quanto il compilatore, vedendo che filter è float, tenta di ottimizzare l'accesso mettendolo tutto nei registri.\
Con 1024 TPB si supera ampiamente il limite di registri per blocco, quindi l'esecuzione termina prematuramente.\
Questo non accadeva nella V1 quando filter era double in quanto il compilatore lo allocava direttamente nello stack locale.\
Per rendere possibili i test con TPB 1024 si fa uso di questo comando di compilazione:
```bash
nvcc -g -G -o main mainV2.cu -DTHREADS_PER_BLOCK=1024 -DTEST_TPB -maxrregcount=48
```
In modo tale da evitare che il compilatore allochi tutto nei registri.\
Infatti facendo così le prestazioni con TPB 1024 sono peggiori.
### V3
Il caso TPB 32 rimane il migliore. Nessuna sincronizzazione tra threads favorisce questo caso in quanto il pù flessibile
dal punto di vista dello scheduler quando assegna blocchi allo SM.
### V4
Il caso TPB 32 rimane il migliore. Nessuna sincronizzazione tra threads favorisce questo caso in quanto il pù flessibile
dal punto di vista dello scheduler quando assegna blocchi allo SM.
### V5
Il caso TPB 256 è in generale il migliore. Per carichi non alti il caso TPB 512 risulta il migliore, ma per garantire più
flessibilità allo scheduler, si opta per TPB 256, la differenza non è sostanziale.
___
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
spostamento dei filtri precalcolati nella memoria costante. Si ha la migliore efficienza quando i thread prelevano lo stesso
filtro contemporaneamente (broadcast hardware).\
Non siamo in grado di garantire una cosa del genere, ma se capita che due threads richiedono lo stesso filtro, allora
migliorano le performance, altrimenti nel caso peggiore si hanno quasi le stesse performance della memoria globale.\
Questa cosa può funzionare in quanto la blurmap non fa mai cambiamenti bruschi nei colori (cerchio sfumato), quindi possiamo
sfruttare la correlazione di valori vicini (pensando alla matrice in quanto matrice, non come vettore).\
Se un thread accede a blurMap\[i]\[j] ed un altro thread accede a blurMap\[i + 1]\[j], è più probabile che trovi lo stesso valore.
Se invece accedono a locazioni lontanissime, sarà meno probabile.

### V4 -> V5
si assegna un intervallo di pixel a livello di blocco, dopodichè i threads si prendono i pixel in maniera alternata:\
thread 0 -> pixel 0\
thread 1 -> pixel 1\
thread 2 -> pixel 2\
...\
thread 255 -> pixel 255\
poi si ricomincia il ciclo con il thread 0 che si prende il pixel 256...\
in questo modo si riescono a fare accessi sequenziali a livello di thread vicini nel tempo.\
I threads di un blocco richiederanno in un certo istante locazioni contigue.
