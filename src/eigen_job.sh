#!/usr/bin/env bash
# File       : eigen_comp.sh
# Description: Determine sequential time & optimized parallel times
#               using OpenMP and AVX2 vectorization
# Copyright 2023 Harvard University. All Rights Reserved.
#SBATCH --job-name=eigen_comp
#SBATCH --output=eigen_comp_outputs/eigen_comp%j.out
#SBATCH --error=eigen_comp_outputs/eigen_comp%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00

export OMP_DYNAMIC='false'
export OMP_PROC_BIND='close'


module purge
module load gcc/12.1.0-fasrc01 openmpi/4.1.3-fasrc01 eigen/3.3.7-fasrc01

make als_eigen
make als_parallel 


echo "Eigen"

for n in 1 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_eigen
done

echo ""
echo "Parallel LU"

for n in 1 2 4 8 16 32
do
    OMP_NUM_THREADS=$n ./als_parallel
done
