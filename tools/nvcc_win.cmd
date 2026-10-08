@echo off
setlocal
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
set "PATH=%CUDA_PATH%\bin;%CUDA_PATH%\libnvvp;%PATH%"

rem CUDA 11.8 needs VS 2017-2022. Prefer VS 2019 BuildTools (MSVC 14.29).
set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not exist "%VCVARS%" (
  echo Missing "%VCVARS%"
  echo Install VS 2019 Build Tools with C++ workload.
  exit /b 1
)
call "%VCVARS%" >nul 2>nul
where cl >nul 2>nul
if errorlevel 1 (
  echo cl.exe not found after calling VS 2019 vcvars64.bat
  exit /b 1
)
where nvcc >nul 2>nul
if errorlevel 1 (
  echo nvcc not found under %CUDA_PATH%\bin
  exit /b 1
)
nvcc -Xcompiler /utf-8 %*
exit /b %ERRORLEVEL%
