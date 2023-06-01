#!/usr/bin/env bash
# File       : cum_job.sh
# Description: Determine sequential time & optimized parallel times
#               using OpenMP and AVX2 vectorization
# Copyright 2023 Harvard University. All Rights Reserved.
#SBATCH --job-name=cum_mf
#SBATCH --output=cum_mf_%j.out
#SBATCH --error=cum_mf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00

export OMP_DYNAMIC='false'
export OMP_PROC_BIND='close'


module purge
module load gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc01 eigen/3.3.7-fasrc01

make clean
make als_parallel
make als_parallel_nosimd
make als_sequential

echo "sequential time: "

./als_sequential

echo "cumulative times:\n "

for n in 1 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel
done
