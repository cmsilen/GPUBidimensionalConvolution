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

REM Compila con nvcc

IF "%~1"=="5" (
    echo compilazione con ottimizzazioni
    nvcc -O3 -use_fast_math -lineinfo -o main %FILE%
    goto fine
)

nvcc -g -G -o main %FILE%

:fine