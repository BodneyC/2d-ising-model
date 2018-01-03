2d Square Lattice Ising Model Simulation
=========================

The purpose of this program is to simulate the two-dimensional square lattice Ising model and use Monte Carlo techniques to estimate state-variables, namely magnetisation and energy. 

There are two versions of the program, the first (`ser_ising.C`) uses OpenMP as its only form of parallelisation and the second (`mpi_ising.C`) uses both the distributed and shared memory models; in honesty, testing has proven that unless the inputs provided to the program are significant, the MPI version is generally slower but it was a good exercise in distributed computing.

The `./scripts/` folder contains general scripts used for testing on a Beowulf-styled cluster and a purpose-built research cluster.

### Usage

First, clone the repo then either `make` the programs or compile manually.

    git clone https://github.com/BodneyC/2dIsingModel.git
    cd 2dIsingModel

then either...

    make

or...

    mpic++ -g -[fq]openmp -O3 -o mpi_ising mpi_ising.C
    (g++|icc) -g -[fq]openmp -O3 -o ser_ising ser_ising.C

### General Notes

- Both versions have been adapted to use global accumaltors for reductions; though this may not be most efficient (in terms of having to reset them at each use), the GCC version available on both clusters (4.8.5) only implements OpenMP 3.1 which does not allow for reductions on reference types (magn[k] etc)

- The serial version certainly compiles on both clusters however I have faced the issue of it not compiling with older versions of gcc as the chessboard decomposition (in do_update()) assigns j based upon i giving:

        error: initializer expression refers to iteration variable ‘i’

, this is not the case with current clang or current versions of gcc.


