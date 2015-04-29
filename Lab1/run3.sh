#!/bin/sh

thread_num=$1
matrix_size=$2

echo "compile application"

gcc -g -fopenmp -march=native -lm matrix3.c -o matrix.exe

echo "setting up number of threads value"
export OMP_NUM_THREADS=$thread_num

echo "executing the application"
./matrix.exe $matrix_size

rm -fr *~ matrix.exe
