/********************************************************************
 * Filename: ser_ising.C [C++ source code]
 *
 * Description: OpenMP implementation of 2D square-lattice Ising model
 *
 * Compilation: (g++|icc) -g -[fq]openmp -O3 -o ser_ising ser_ising.C
 *
 * Author: Biagio Lucini (Optimised by Benjamin Carrington)
 *
 *******************************************************************/

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include "omp.h"

#define NR_RNDNUM 365
#define NUM_BINS 25 
#define NUM_THREADS 8
#define quick_and_dirty_rand() rand()/double(RAND_MAX)

/** Function prototypes */
void read_input(void);
void print_log(void);
double rndnum(int* icall, int* iv);
void init_spin(int chunk_width, int* icall, int* iv);
void rini(int iran, int* iv);
void initweight(void);
int neighbour(int a, int b);
int staple(int a, int b);
int Heatbath(int a, int b, int* icall, int* iv);
void do_measurements(int a, int b);
void write_measures(double elapsed);
void jack_error(double *obs, double *avg, double *err);
void do_update(int chunk_width, int* icall, int* iv);

/** Variable definitions */
int **spin;
int thermalisation;
int size;
int measurements;
int istart;
double beta;
double inv_beta;
double weight[5];               
double volume;
double *magn, *energy;
std::ofstream outfile("measures.dat", std::ios_base::app); 

/** Necessary for GCC versions available in labs */
double tmp_accum_1, tmp_accum_2;

/****************************** MAIN *******************************/

int main(int argc, char** argv)
{
    unsigned int randomseed = 1812; 
    double avg_energy = 0, err_energy = 0;
    double avg_magn = 0, err_magn = 0;
    struct timespec start, finish;
    double elapsed;
    int chunk_width;
    int ran_arr[NUM_THREADS];
    omp_set_num_threads(NUM_THREADS);

    /** Seed rand() */
    srand(randomseed);

    /** Generate random number for rini() per thread */
    for(int i = 0; i < NUM_THREADS; i++)
        ran_arr[i] = int(quick_and_dirty_rand() * 259200);

    /** stdin and log */
    read_input();
    print_log();
    inv_beta = 1.0 / beta;

    /** Pre-threaded sanity checks */
    if ((measurements / NUM_BINS * NUM_BINS) != measurements)
    {
        std::cout << "Error: the number of measurements must be divisible by NUM_BINS" << std::endl;
        std::cout << "We have NUM_BINS = " << NUM_BINS <<  " and " << measurements  << " requested measurements" << std::endl;
        exit(EXIT_FAILURE);
    }

	if(size & 1)
	{
		printf("Size must be even for chessboard dependency purposes.\n  Size:\t%d\n", size);
		exit(EXIT_FAILURE);
	}

    /** Initialisation of observables */
    energy = new double [measurements]();
    magn = new double [measurements]();
    volume = static_cast<double>(size) * static_cast<double>(size);
    spin = new int* [size];
    for(int i = 0; i < size; i++)
        spin[i] = new int [size];

    /** Start clock */
    clock_gettime(CLOCK_MONOTONIC, &start);

    initweight();

#pragma omp parallel default(shared)
{
    if(NUM_THREADS != omp_get_num_threads())
    {
        printf("Thread creation issue.\n  #Threads requested:\t%d\n"
                "  #Threads spawned:\t%d\n", NUM_THREADS, omp_get_num_threads());
        exit(EXIT_FAILURE);
    }

    int thread_num = omp_get_thread_num();
    int ind_seed = ran_arr[thread_num];
    int icall = 0;
    int iv[NR_RNDNUM] = {0};

#pragma omp single
    chunk_width = size / NUM_THREADS;

    if(chunk_width * NUM_THREADS != size)
    {
        printf("Number of threads (%d) must divide into lattice width (%d).\n", NUM_THREADS, size);
        exit(EXIT_FAILURE);
    }

    /** Initialise rndnum() per-thread */
    rini(ind_seed, iv);
    init_spin(chunk_width, &icall, iv);

    /** Thermalisaion */
    for(int i = 0; i < thermalisation; i++)
        do_update(chunk_width, &icall, iv);

    /** Measurements */
    for(int i = 0; i < measurements; i++)
    {
        do_update(chunk_width, &icall, iv);
        do_measurements(i, chunk_width);
    }

    /** Jackknife error calculations */
    jack_error(energy, &avg_energy, &err_energy);
    jack_error(magn, &avg_magn, &err_magn);
}

    /** Finish clock and calc elapsed time*/
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    /** Stdout and write measurements */
    std::cout << "Avg energy : " << avg_energy << " +/- " << err_energy << std::endl;
    std::cout << "Avg magn: " << avg_magn << " +/- " << err_magn << std::endl;

    write_measures(elapsed);

    /** Free memory */
    delete[] energy;
    delete[] magn;
    for(int i =0; i < size; i++)
        delete[] spin[i];
    delete[] spin;

    return 0;
}

/****************************** DO UPDATE *******************************/

void do_update(int chunk_width, int* icall, int* iv)
{
    /** White squares */
#pragma omp for collapse(2) schedule(static, chunk_width)
    for (int i = 0; i < size; i++)
        for (int j = i & 1; j < size; j += 2 )
            spin[i][j] =  Heatbath(i, j, icall, iv);

    /** Black squares */
#pragma omp for collapse(2) schedule(static, chunk_width)
    for (int i = 0; i < size; i++)
        for (int j = !(i & 1); j < size; j += 2 )
            spin[i][j] =  Heatbath(i, j, icall, iv);
}

/****************************** DO MEASUREMENTS *******************************/

void do_measurements(int k, int chunk_width)
{
#pragma omp single
    tmp_accum_1 = tmp_accum_2 = 0.;        

#pragma omp for collapse(2) schedule(static, chunk_width) reduction(+: tmp_accum_1, tmp_accum_2)
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j ++ )
        {
            int ifor = (i + 1) % size;
            int jfor = (j + 1) % size;
            tmp_accum_1 += spin[i][j];
            tmp_accum_2 += spin[i][j] * (spin[ifor][j] + spin[i][jfor]);
        }

#pragma omp single nowait
    magn[k] = fabs(tmp_accum_1) / volume;

#pragma omp single
    energy[k] = tmp_accum_2 / volume;
}

/****************************** INIT SPIN *******************************/

void init_spin(int chunk_width, int* icall, int* iv)
{
#pragma omp for collapse(2) schedule(static, chunk_width) 
    for(int i = 0; i < size; i++)
        for(int j = 0; j < size; j++)
            switch (istart)
            {
            case 0:
                spin[i][j] = 1;
                break;
            case 1:
                spin[i][j] = rndnum(icall, iv) < 0.5 ? 1 : -1;
                break;
            default:
                std::cout << "Unknown initialisation parameter" << std::endl;
                exit(EXIT_FAILURE);
            }
}

/****************************** JACK ERROR *******************************/

void jack_error(double *obs, double *avg, double *err)
{
    int slice = measurements / NUM_BINS;
    double* bin = new double [NUM_BINS]();
    double* jackbins = new double [NUM_BINS]();
    double sumbins;

#pragma omp single
    tmp_accum_1 = 0.;

#pragma omp for schedule(static) reduction(+: tmp_accum_1)
    for (int l = 0; l < measurements; l++)
        tmp_accum_1 += obs[l];

#pragma omp single
{
    *avg = tmp_accum_1 / measurements;
    tmp_accum_1 = 0;
}

#pragma omp for schedule(static) reduction(+: tmp_accum_1)
    for (int l = 0; l < NUM_BINS; l++)
    {
        for (int k = 0; k < slice; k++)
            bin[l] += obs[l * slice + k];

        bin[l] /= slice;
        tmp_accum_1 += bin[l];
    }

	sumbins = tmp_accum_1;

#pragma omp single
    tmp_accum_1 = 0;

#pragma omp for schedule(static) reduction(+: tmp_accum_1)
    for (int l = 0; l < NUM_BINS; l++)
	{
        jackbins[l] = (sumbins - bin[l]) / (NUM_BINS - 1);
        tmp_accum_1 += (*avg - jackbins[l]) * (*avg - jackbins[l]);
	}

#pragma omp single
{
    *err = tmp_accum_1 * ((static_cast<double>(NUM_BINS - 1)) / static_cast<double>(NUM_BINS));
    *err = sqrt(*err);
}

    delete[] bin;
    delete[] jackbins;
}

/****************************** STAPLE *******************************/

int staple (int i, int j)
{
    int ifor = (i + 1) % size;
    int jfor = (j + 1) % size;
    int iback = ((i - 1) + size) % size;
    int jback = ((j - 1) + size) % size;

    return spin[ifor][j] + spin[iback][j] + spin[i][jfor] + spin[i][jback];
}

/****************************** HEATBATH *******************************/

int Heatbath(int i, int j, int* icall, int* iv)
{
    int trialspin;
    int k = staple(i, j);
    int m = k / 2 + 2;
    double rran = rndnum(icall, iv);

    if (rran < weight[m])
        trialspin = -1;
    else
        trialspin = 1;

    return trialspin;
}

/****************************** INIT WEIGHT *******************************/

void initweight ()
{
    for(int i = 0; i < 5; i++)
    {
        int j = 2 * (i - 2);
        weight[i] = 1.0 / (1.0 + exp(2 * j * beta));
    }
}

/****************************** RNDNUM *******************************/

double rndnum(int* icall_o, int* iv)
{
    int ivn;
    double frndnum;
    int ir = 24;
    int is = 10;
    int irs = ir - is;
    int ir1 = ir - 1;
    int ikeep = ir;
    int ithrow = 24;			// normal
    int ivl1 = ikeep + ithrow;
    long ibase = long(pow(2, 24));
    double basein = 1.0 / ibase;

    int icall = *icall_o;

    if (icall == ikeep)
    {
        for(int i = 1; i <= ithrow; i++)
        {
            ivn = iv[irs + icall] - iv[icall];
            icall++;

            if (ivn < 0)
            {
                iv[icall]++;
                ivn = ivn + ibase;
            }

            iv[icall + ir1] = ivn;
        }

        for(int i = 0; i <= ir1; i++)
            iv[i] = iv[ivl1 + i];

        icall = 0;
    }

    ivn = iv[irs + icall] - iv[icall];
    icall++;

    if (ivn < 0)
    {
        iv[icall]++;
        ivn = ivn + ibase;
    }

    iv[icall + ir1] = ivn;

    *icall_o = icall;

    frndnum = double(ivn) * basein;
    return(1. - frndnum);
}

/****************************** RINI *******************************/

void rini(int iran, int* iv)
{
    int jran, ifac;
    int ir = 24;
    int im = 259200;
    int ia = 7141;
    int ic = 54773;
    long ibase = long(pow(2, 24));

    printf("Thread rank: %d - Random seed: %d \n", omp_get_thread_num(), iran);

    if((iran < 0) || (iran >= im))
        iran = im - 1;

    jran = iran;

    for (int i = 0; i < 10; i++)
        jran = (jran * ia + ic) % im;

    ifac = (ibase - 1) / im;

    for(int i = 0; i < ir; i++)
    {
        jran = (jran * ia + ic) % im;
        iv[i] = ifac * jran;
    }
}

/****************************** READ INPUT *******************************/

void read_input()
{
    std::cin >> size;
    std::cin >> beta;
    std::cin >> thermalisation;
    std::cin >> measurements;
    std::cin >> istart;

    if ( (istart != 1) && (istart != 0) )
    {
        std::cout << "Error: istart can only be either 0 (cold) or 1 (hot)" << std::endl;
        exit(EXIT_FAILURE);
    }
}

/****************************** PRINT LOG *******************************/

void print_log()
{
    std::cout << "Ising model on a 2D square lattice" << std::endl;
    std::cout << "Size of the lattice: " << size << std::endl;
    std::cout << "Beta: " << beta << std::endl;
    std::cout << "Thermalisation sweeps: " << thermalisation << std::endl;
    std::cout << "Number of measurements: " << measurements << std::endl;

    if (istart == 1)
    {
        std::cout << "Start: hot" << std::endl;
    }
    else
    {
        std::cout << "Start: cold" << std::endl;
    }
}

/****************************** WRITE MEASURES *******************************/

void write_measures(double elapsed)
{
    outfile << size << "\t" <<  elapsed << "\n";
    //for(int i = 0; i < 1; i ++)
    //    outfile << inv_beta << "\t" << energy[i] << "\t" << energy[i] * energy[i] << "\t" << magn[i] << "\t" << magn[i] * magn[i] << "\n";
}

