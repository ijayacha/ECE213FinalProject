#!/bin/bash
set -e   # stop on first error — helps catch issues early

cd /home/ijayacha/ECE213FinalProject/GHMMGenePredictor   # make sure we're in the right directory, update path to your working directory

# Compile
nvcc -O3 -arch=sm_80 viterbi.cu -o viterbi -lm

# Benchmark sweep across sequence lengths
for T in 10000 50000 100000 500000 1000000; do
    echo "=== T=$T ==="
    ./viterbi $T
done