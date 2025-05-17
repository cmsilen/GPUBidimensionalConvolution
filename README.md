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
