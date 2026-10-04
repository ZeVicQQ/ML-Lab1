@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Запуск лабораторной работы №1
echo ============================================

REM --- Проверка Python ---
where python >nul 2>nul
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Python не найден.
    echo Установите Python 3.10+ с https://www.python.org/downloads/
    echo При установке отметьте "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

REM --- Создание виртуального окружения, если нет ---
if not exist ".venv\Scripts\python.exe" (
    echo Создание виртуального окружения .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось создать venv.
        pause
        exit /b 1
    )
)

REM --- Установка зависимостей, если не установлены ---
".venv\Scripts\python.exe" -c "import pandas, numpy, sklearn, openpyxl" >nul 2>nul
if errorlevel 1 (
    echo Установка зависимостей...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ОШИБКА] Не удалось установить зависимости.
        pause
        exit /b 1
    )
)

REM --- Запуск ---
echo.
echo Запуск lab1.py ...
".venv\Scripts\python.exe" lab1.py

echo.
echo Готово. Результаты в папке lab1_results.
pause