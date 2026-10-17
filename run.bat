@echo off
rem VectorDB Java Server - Windows run script (double-click or run from cmd/PowerShell)
rem Requires: Java 21+. Optional argument: port (default 9999)
setlocal enabledelayedexpansion
cd /d "%~dp0"
echo === VectorDB Java Server ===

rem Find a Java 21+ runtime: PATH first, then JAVA_HOME, then common install folders
set "JAVA="
call :try_java "java"
if not defined JAVA if defined JAVA_HOME call :try_java "%JAVA_HOME%\bin\java.exe"
for /d %%d in ("%ProgramFiles%\Java\jdk-*" "%ProgramFiles%\Microsoft\jdk-*" "%ProgramFiles%\Eclipse Adoptium\jdk-*" "%ProgramFiles%\Zulu\zulu-*" "%ProgramFiles%\Amazon Corretto\jdk*") do (
    if not defined JAVA call :try_java "%%~d\bin\java.exe"
)
if not defined JAVA (
    echo.
    echo Java 21 or newer is required but was not found.
    echo Install it from https://adoptium.net then run this again.
    pause
    exit /b 1
)
echo Using Java %JAVA_MAJOR%: %JAVA%

rem Rebuild the jar from source if it is missing
if not exist vectordb.jar (
    echo Compiling from source ...
    if "%JAVA%"=="java" (set "BIN=") else (for %%f in ("%JAVA%") do set "BIN=%%~dpf")
    if exist out rmdir /s /q out
    "!BIN!javac" -d out *.java || (echo Compilation failed. & pause & exit /b 1)
    "!BIN!jar" cfe vectordb.jar com.vectordb.VectorDBServer -C out . || (echo Packaging failed. & pause & exit /b 1)
)

"%JAVA%" -jar vectordb.jar %*
if errorlevel 1 pause
exit /b

:try_java
rem Sets JAVA and JAVA_MAJOR if %1 is a Java 21+ executable
if /i not "%~1"=="java" if not exist "%~1" exit /b
"%~1" -version 2>"%TEMP%\vectordb-java-version.txt" >nul || exit /b
set "VER="
for /f "usebackq tokens=3" %%v in (`findstr /i "version" "%TEMP%\vectordb-java-version.txt"`) do set "VER=%%~v"
set "MAJOR=0"
for /f "delims=.-+" %%m in ("!VER!") do set "MAJOR=%%m"
if !MAJOR! GEQ 21 (
    set "JAVA=%~1"
    set "JAVA_MAJOR=!MAJOR!"
)
exit /b
