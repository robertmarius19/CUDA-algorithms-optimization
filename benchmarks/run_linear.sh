#!/bin/bash
set -e

OUT="linear_results.csv"
echo "algo,version,N,time_ms" > "$OUT"

DFT_BIN="../DFT"
GEMM_BIN="../GEMM"

DFT_SIZES=(512 1024 2048 4096 8192 16384 32768)
GEMM_SIZES=(512 1024 2048 4096 8192 16384)

# DFT
for N in "${DFT_SIZES[@]}"; do
  echo "Running DFT N=$N"

  # sequential
  TIME=$($DFT_BIN/dft_secv $N)
  echo "DFT,seq,$N,$TIME" >> "$OUT"

  # omp
  TIME=$($DFT_BIN/dft_cpu $N)
  echo "DFT,omp,$N,$TIME" >> "$OUT"

  # cuda
  TIME=$($DFT_BIN/dft_cuda $N)
  echo "DFT,cuda,$N,$TIME" >> "$OUT"
  echo "---------------------">> "$OUT"
done

# GEMM
for N in "${GEMM_SIZES[@]}"; do
  echo "Running GEMM N=$N"

  if [ $N -le 2048 ]; then
    TIME=$($GEMM_BIN/gemm_secv $N)
    echo "GEMM,seq,$N,$TIME" >> "$OUT"
  fi

  TIME=$($GEMM_BIN/gemm_cpu $N)
  echo "GEMM,omp,$N,$TIME" >> "$OUT"

  TIME=$($GEMM_BIN/gemm_cuda $N)
  echo "GEMM,cuda,$N,$TIME" >> "$OUT"
  echo "---------------------">> "$OUT"
done
