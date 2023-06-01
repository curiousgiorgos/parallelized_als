#!/usr/bin/env bash
# File       : simd_trials_job.sh
# Description: Determine sequential time & optimized parallel times
#               using OpenMP and AVX2 vectorization
# Copyright 2023 Harvard University. All Rights Reserved.
#SBATCH --job-name=simd_trials_job
#SBATCH --output=simd_trials_outputs/simd_trials_job%j.out
#SBATCH --error=simd_trials_outputs/simd_trials_job%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

export OMP_DYNAMIC='false'
export OMP_PROC_BIND='close'

module purge
module load gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc01 eigen/3.3.7-fasrc01

make clean
make als_simd

echo "Full:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_full
done

echo ""
echo "No FMA:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_nofma
done

echo ""
echo "No SIMD:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_nosimd
done

echo ""
echo "Single FMA:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_singlefma
done
