#!/bin/sh

for THR in 4 8
do
    sed "s/NUM_THREADS [0-9]\+/NUM_THREADS ${THR}/" ser_ising.C > ser_tmp.c && mv ser_tmp.c ser_ising.C

    make clean && make

    for BETA in `cat beta.list`
    do
        for WIDTH in `cat width.list`
        do
            sed "s/DIM/$WIDTH/g" input.prov | sed "s/BETA/$BETA/g" > input

            ./ser_ising < input
        done

        mv measures.dat FULL_SER_${THR}_${BETA}.dat
    done		
done

