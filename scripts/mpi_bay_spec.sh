#!/bin/sh

#SBATCH --job-name="BJC-Ising-MPI"
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=1024
#SBATCH -o ISING.out
#SBATCH -e ISING.err
#SBATCH -n 14
#SBATCH -c 2

if [ -n "$SLURM_CPUS_PER_TASK" ]
then
    omp_threads=$SLURM_CPUS_PER_TASK
else 
    omp_threads=1 
fi 

export OMP_NUM_THREADS=$omp_threads

module load intel
module load mvapich2

mpiexec ./mpi_ising < input > out

