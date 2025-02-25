@echo off
REM Build script for Lucyna project
REM This script runs CMake, builds all targets, and runs all tests

echo ===== Building Lucyna project =====

REM Create build directory if it doesn't exist
if not exist "build" mkdir build
cd build

echo.
echo ===== Running CMake =====
cmake ..

echo.
echo ===== Building all targets =====
cmake --build . --config Release

cd ..

echo.
echo ===== Running tests =====

cd bin

echo.
echo ----- Running Tokenizer Test -----
lucyna_test_tokenizer.exe

echo.
echo ----- Running Transformer Test -----
lucyna_test_transformer.exe

echo.
echo ----- Running Tensor Test -----
lucyna_test_tensor.exe

echo.
echo ----- Running Tensor Math Test -----
lucyna_test_tensor_math.exe

echo.
echo ----- Running Attention Test -----
lucyna_test_attention.exe

echo.
echo ----- Running BFloat16 Test -----
lucyna_test_bfloat16.exe

echo.
echo ===== All tests completed =====