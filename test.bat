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


for /L %%i in (1,1,60) do (
    for %%j in (1,5,10) do (
        echo Esecuzione: main %%i threads %%j imgs
        main %%i %%j 7 0 1
        main %%i %%j 7 0 1
        main %%i %%j 7 0 1
        main %%i %%j 7 0 1
        main %%i %%j 7 0 1
    )
)

:fine