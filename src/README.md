## Prerequisites 

gcc/12.1.0
\
Install on cluster using: 'module load gcc/12.1.0-fasrc01' 

openmpit/4.1.3
\
Install on cluster using: 'module load openmpi/4.1.3-fasrc01'

eigen/3.3.7
\
Install on cluster using: 'module load eigen/3.3.7-fasrc01'

## Usage 

The relavant scripts are include under the /src repository. 

The Makefile allows you to build different versions of the ALS algorithm with different settings and configurations. Here are the steps to use the Makefile:

1. Ensure that the relevant libraries outlined on the prerequisites section are installed 
2. Open a terminal window and naviate to the /src directory
3. Type 'make main' in the terminal to build the: als_sequential and als_parallel scripts 
4. The executables can be run by using './als_parallel' and './als_sequential'. You can use 'OMP_NUM_THREADS=n  ./als_parallel 'to run the parallel executable with n threads
5. Alternatively you can use 'make' to all targets
6. As above you can use './script' to run the relevant executable and OMP_NUM_THREADS to specify the number of threads
7. run 'make clean' to remove the executables

Running als_parallel and als_sequential will execute until the end of the algorithm when the total runtime of the ALS portion would be outputed on the terminal. 

Alternatively, if you are viewing the repository on the academic cluster you can use sbatch to submit a job with on of the job scripts: 

1. 'sbatch eigen_job.sh': compares the performance of eigen to the SIMD parallel als implementation, the results will be outputed under simd_tirals_outputs
2. 'sbatch loopsched.sh': compares the performance of diffrent loop scheduling techniques on non vectorized code and outputs the results under loopsched_job_outputs
3. 'sbatch simd_trials_job.sh': compares the performance of diffrent loop SIMD implementations and outputs the results under simd_trials_outputs
4. 'sbatch cum_job.sh': compares the performance als_sequential and als_parallel for different number of threads and outputs the results under cum_job_outpputs

## Reproducibility 

The benchamarks were run on the Harvard FAS-Academic Cluster using an Intel XEON E5-2683v4 CPU.

The resources required were 1 node with 32 cores. In order to reproduce the results as closely as possible we recommend that you use the same resources

All the relavant data sets used are listes under the /data repository