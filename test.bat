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
call compile %~1


for %%i in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048) do (
    for %%j in (3,15,30) do (
        echo Esecuzione: main %%i blocks %%j imgs
        main %%i %%j 1
        main %%i %%j 1
        main %%i %%j 1
    )
)

:fine