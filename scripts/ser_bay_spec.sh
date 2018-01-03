#!/bin/sh

#SBATCH --job-name="BJC-Ising-Ser"
#SBATCH -o ISING.out
#SBATCH -e ISING.err
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=1024
#SBATCH --ntasks=28

module load intel
module load mvapich2

for BETA in `cat beta.list`
do
    for DIM in `cat width.list`
    do
        sed "s/DIM/$DIM/" input.prov | sed "s/BETA/$BETA/" > input
        
        ./ser_ising < input > output

    done
    mv measures.dat ${BETA}_FULL.dat
done

