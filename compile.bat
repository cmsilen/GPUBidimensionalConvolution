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

REM Compila con gcc

IF "%~1"=="5" (
    echo compilazione con ottimizzazioni matematiche
    nvcc -o main %FILE%
    goto fine
)

nvcc -g -G -o main %FILE%

:fine