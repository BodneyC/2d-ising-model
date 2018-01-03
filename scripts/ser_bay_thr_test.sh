#!/bin/sh

for i in 1 2 4 8 16 28
do
    sed "s/ntasks=[0-9]\+/ntasks=${i}/" ser_bay_spec.sh > tmp_job_sub && mv tmp_job_sub ser_bay_spec.sh
    sed "s/NUM_THREADS [0-9]\+/NUM_THREADS ${i}/" ser_ising.C > tmp_isi && mv tmp_isi ser_ising.C

    icc -o ser_ising ser_ising.C -qopenmp -g -O3

    sbatch job-spec-ser.sh

    sleep 2
    QUEUE=`squeue -u 927772 | grep 927772`

    while [[ ! -z $QUEUE ]]
    do
        sleep 2
        QUEUE=`squeue -u 927772 | grep 927772`
    done
done

