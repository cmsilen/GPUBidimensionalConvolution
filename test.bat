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

IF "%~1"=="1" (
    for %%i in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384) do (
        for %%j in (3,15,30) do (
            echo Esecuzione: main %%i blocks %%j imgs
            main %%i %%j 1
        )
    )
    goto fine
)

IF "%~1"=="5_HighWorkload" (
    for %%i in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768) do (
        for %%j in (30,100,200) do (
            echo Esecuzione: main %%i blocks %%j imgs
            main %%i %%j 1
            main %%i %%j 1
            main %%i %%j 1
        )
    )
    goto fine
)

for %%i in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384) do (
    for %%j in (3,15,30) do (
        echo Esecuzione: main %%i blocks %%j imgs
        main %%i %%j 1
        main %%i %%j 1
        main %%i %%j 1
    )
)

:fine