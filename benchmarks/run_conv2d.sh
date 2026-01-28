#!/bin/bash
set -e

OUT="conv2d_results.csv"
echo "algo,version,N,time_ms" > "$OUT"

CONV_BIN="../Convolutie2D"

SIZES=(256 512 1024 2048 4096 8192 16384)

echo "Starting Conv2d Benchmarks..."

for N in "${SIZES[@]}"; do
  echo "Running Conv2d N=$N"


  
  TIME=$($CONV_BIN/conv2d_secv $N)
  echo "CONV,seq,$N,$TIME" >> "$OUT"
  

  # 2. OpenMP
  TIME=$($CONV_BIN/conv2d_cpu $N)
  echo "CONV,omp,$N,$TIME" >> "$OUT"

  # 3. CUDA
  TIME=$($CONV_BIN/conv2d_cuda $N)
  echo "CONV,cuda,$N,$TIME" >> "$OUT"
  
  echo "---------------------" >> "$OUT"
done

echo "Done. Results saved to $OUT"