@echo off

REM Controllo se è stato passato un parametro
IF "%~1"=="" (
    echo Uso: compila.bat ^<numero^>
    goto fine
)

REM Prende il numero in input
set NUM=%1

REM Costruisce il nome del file
set FILE=mainV%NUM%.cu

REM pulizia file vecchi
del main.exe
del main.exp
del main.lib
del main.pdb
del vc140.pdb
echo pulizia completata

for %%i in (32, 64, 128, 256, 512, 1024) do (
    nvcc -g -G -o main %FILE% -DTHREADS_PER_BLOCK=%%i -DTEST_TPB
    for %%j in (3,15,30) do (
        echo Esecuzione: main %%i thread per block %%j imgs
        main 1 %%j 1
        main 1 %%j 1
        main 1 %%j 1
    )
)