#!/bin/sh

for i in `cat beta.list`
do
    for L in `cat width.list`
    do
        sed "s/DIM/${L}/g" input.prov | sed "s/BETA/${i}/g" > input
        #sed "s/ntasks=[0-9]\+/ntasks=${TASKS}/g" job-spec-mpi.sh \
        #   > job-mpi.TMP && mv job-mpi.TMP job-spec-mpi.sh

        sbatch job-spec-mpi.sh

        QUEUE=`squeue -u 927772 | grep 927772`

        while [[ ! -z $QUEUE ]]
        do
            sleep 2
            QUEUE=`squeue -u 927772 | grep 927772`
        done
    done		
    mv measures.dat FULL_${i}.dat
done

rm measures.dat

