#!/usr/bin/env bash
# File       : loopsched_job.sh
# Description: Determine sequential time & optimized parallel times
#               using OpenMP and AVX2 vectorization
# Copyright 2023 Harvard University. All Rights Reserved.
#SBATCH --job-name=loopsched_job
#SBATCH --output=loopsched_job_outputs/loopsched_job%j.out
#SBATCH --error=loopsched_job_outputs/loopsched_job%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00

export OMP_DYNAMIC='false'
export OMP_PROC_BIND='close'


module purge
module load gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc01 eigen/3.3.7-fasrc01

make clean
make als_loopsched


echo "Dynamic 16:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_dynamic16
done

echo ""
echo "Dynamic 32:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_dynamic32
done

echo ""
echo "Dynamic 64:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_dynamic64
done

echo ""
echo "Dynamic 128:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_dynamic128
done

echo ""
echo "Guided:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_guided
done

echo ""
echo "Static:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_static
done

echo ""
echo "Static 2:"

for n in 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel_static2
done


