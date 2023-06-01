#!/usr/bin/env bash
# File       : schedule_job.sh
# Description: Determine sequential time & optimized parallel times
#               using OpenMP with different schedules
# Copyright 2023 Harvard University. All Rights Reserved.
#SBATCH --job-name=schedule_mf
#SBATCH --output=schedule_mf_%j.out
#SBATCH --error=schedule_mf_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00

export OMP_DYNAMIC='false'
export OMP_PROC_BIND='close'


module purge
module load gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc01 eigen/3.3.7-fasrc01

make clean
make als_parallel_static
make als_parallel_dynamic2
make als_parallel_dynamic128
make als_parallel_guided

echo "static times:\n "

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_static
done

echo "dynamic_2 times:\n "

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_dynamic2
done

echo "dynamic_128 times:\n "

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_dynamic128
done

echo "guided_16 times:\n "

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_guided
done