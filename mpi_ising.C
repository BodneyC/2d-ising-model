/********************************************************************
 * Filename: mpi_ising.C [C++ source code]
 *
 * Description: OpenMP and MPI hybrid implementation of 2D square-
 *              lattice Ising model
 *
 * Compilation: mpic++ -g -[fq]openmp -O3 -o mpi_ising mpi_ising.C
 *
 * Author: Original - Biagio Lucini 
 *         Optimised - Benjamin Carrington
 *
 *******************************************************************/

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include "omp.h"

#define NR_RNDNUM 365
#define quick_and_dirty_rand() rand()/double(RAND_MAX)
#define NUM_THREADS 2

/** Function prototypes */
void read_input(void);
void print_log(void);
double rndnum(int* icall, int* iv);
void init_spin_MPI(int* icall, int* iv);
void rini(int iran, int* iv);
void initweight(void);
int neighbour(int a, int b);
int staple(int a, int b);
int Heatbath(int a, int b, int* icall, int* iv);
void do_update_bulk_MPI(int* icall, int* iv);
void do_update_row1_MPI(int* icall, int* iv);
void do_updatetot_MPI(int* icall, int* iv, int up_id, int dn_id, int myparity);
void do_measurements_MPI(int a);
void write_measures_MPI(double elapsed);

/** Variable definitions */
int **spin;
unsigned int thermalisation;
unsigned int size, sizenp, sizenp1, sizenp2;
unsigned int measurements;
int istart;
double beta;
double inv_beta;
double weight[5];
double volume, volumenp;
double *magn, *ene, *magntot, *enetot;
std::ofstream outfile("measures.dat", std::ios_base::app); // While recording time
//std::ofstream outfile("measures.dat"); // While outputting orig
int num_procs;
int chunk_width;
MPI_Status status;
int ran_arr[NUM_THREADS];

/** Necessary for GCC versions available in labs */
double tmp_accum_1, tmp_accum_2;

/****************************** MAIN *******************************/

int main(int argc, char *argv[])
{
    int proc_id;
    double start, finish, elapsed;
    int up_id, dn_id, myparity;
    int randomseed = 1812;

    /** Init MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
    omp_set_num_threads(NUM_THREADS);

    /** stdin and log */
    if(proc_id == 0)
    {
        read_input();
        print_log();
    }

    MPI_Bcast(&size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&thermalisation, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&measurements, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    sizenp = size / num_procs;
    sizenp1 = sizenp + 1;
    sizenp2 = sizenp + 2;

    /** Seed rand() per process */
    srand(randomseed * (1 + proc_id));

    /** Generate random number array before OpenMP env */
    for(int i = 0; i < NUM_THREADS; i++)
        ran_arr[i] = int(quick_and_dirty_rand() * 259200);

    /** MPI specific variables */
    up_id = (proc_id + 1) % num_procs; // Up rankID
    dn_id = (proc_id + num_procs - 1) % num_procs; // down rankID
    myparity = proc_id & 1;
    if(num_procs == 1)
        myparity = 3;

    /** Initialisaddtion of observables */
    ene = new double [measurements]();
    enetot = new double [measurements];
    magn = new double [measurements]();
    magntot = new double [measurements];
    volume = (double)size * (double)size;
    volumenp = (double)size * (double)sizenp;

    /** Allocate lattice */
    spin = new int* [sizenp2]; 
    for(int i = 1; i < sizenp1; i++)
        spin[i] = new int [size]; 
    spin[0] = new int [size]();
    spin[sizenp1] = new int [size]();

    initweight();

    /** Pre-OpenMP sanity checks */
    if(size & 1)
    {
        printf("Error, size must be even for chessboard dependency.\n");
        exit(EXIT_FAILURE);
    }
    if(sizenp * num_procs != size)
    {
        printf("Error! size has to be multiple of num_procs\n");
        exit(EXIT_FAILURE);
    }

#pragma omp parallel default(shared) 
{
    if(NUM_THREADS != omp_get_num_threads())
    {
        printf("Thread creation issue.\n  #Threads requested:\t%d"
                "\n  #Threads spawned:\t%d\n", NUM_THREADS, omp_get_num_threads());
        exit(EXIT_FAILURE);
    }

    int thread_num = omp_get_thread_num();
    int icall = 0;
    int	iv[NR_RNDNUM] = {0};
    int ind_seed = ran_arr[thread_num];

#pragma omp single
    chunk_width = size / NUM_THREADS;

    if(chunk_width * NUM_THREADS != size)
    {
        printf("Number of threads (%d) must divide into lattice width (%d)\n", NUM_THREADS, size);
        exit(EXIT_FAILURE);
    }

    /** Initialise rndnum() per thread */
    rini(ind_seed, iv);
    init_spin_MPI(&icall, iv); 

#pragma omp master
    start = MPI_Wtime();
#pragma omp barrier

    /** Thermalisation */
    for (int i = 0 ; i < thermalisation ; i++)
        do_updatetot_MPI(&icall, iv, up_id, dn_id, myparity);


    /** Measurements */
    for (int i = 0 ; i < measurements ; i++)
    {
        do_updatetot_MPI(&icall, iv, up_id, dn_id, myparity);
        do_measurements_MPI(i);
    }
}

    /** Reduce per-process observables */
    MPI_Reduce(magn,
            magntot,
            measurements,
            MPI_DOUBLE, 
            MPI_SUM,   
            0,
            MPI_COMM_WORLD
            );

    MPI_Reduce(ene, 
            enetot, 
            measurements, 
            MPI_DOUBLE, 
            MPI_SUM, 
            0,
            MPI_COMM_WORLD
            );

    /** Calc time and write measurements */
    if(proc_id == 0)
    {
        finish = MPI_Wtime();
        elapsed = finish - start;
        inv_beta = 1.0 / beta;

        write_measures_MPI(elapsed);
    }

    /** Free memory and finalise environment */
    delete[] ene;
    delete[] enetot;
    delete[] magn;
    delete[] magntot;
    for(int i = 0; i < sizenp2; i++)
        delete[] spin[i];
    delete[] spin;
    MPI_Finalize();

    return 0;
}

/************************** INIT SPIN MPI **************************/

void init_spin_MPI(int* icall, int* iv)
{
#pragma omp for schedule(static, chunk_width)
    for(int j = 0; j < size; j++)
        for(int i = 1; i <= sizenp; i++)
            switch (istart)
            {
            case 0:
                spin[i][j] = 1 ;
                break;
            case 1:
                spin[i][j] = rndnum(icall, iv) < 0.5 ? 1 : -1;
                break;
            default:
                std::cout << "Unknown initialisation parameter" << std::endl;
                exit(EXIT_FAILURE);
                break;
            }
}


/*************************** MEASUREMENTS **************************/

void do_measurements_MPI(int k)
{
#pragma omp single
    tmp_accum_1 = tmp_accum_2 = 0.;

#pragma omp for schedule(static, chunk_width) reduction(+: tmp_accum_1, tmp_accum_2)
    for (int j = 0 ; j < size ; j++ )
        for (int i = 1 ; i < sizenp1 ; i++)
        {
            int ifor = i + 1;
            int jfor = (j + 1) % size;
            tmp_accum_1 += spin[i][j] ;
            tmp_accum_2 += spin[i][j] * (spin[ifor][j] + spin[i][jfor]);
        }

#pragma omp single nowait
{
    magn[k] = fabs(tmp_accum_1) / volumenp ;
    magn[k] /= num_procs;
}
#pragma omp single 
{
    ene[k] = tmp_accum_2 / volumenp ;
    ene[k] /= num_procs;
}
}

/**************************** UPDATE: 2 ****************************/

void do_update_bulk_MPI(int* icall, int* iv)
{
#pragma omp for schedule(static, chunk_width) 
    for(int j = 0; j < size; j++)
        for(int i = (j & 1) + 2; i <= sizenp; i+=2)
            spin[i][j] =  Heatbath(i, j, icall, iv) ;
    
#pragma omp for schedule(static, chunk_width) 
    for(int j = 0; j < size; j++)
        for(int i = !(j & 1) + 2; i <= sizenp; i+=2)
            spin[i][j] =  Heatbath(i, j, icall, iv) ;
}

/**************************** UPDATE: 3 ****************************/

void do_update_row1_MPI(int* icall, int* iv)
{
    for (int j = 0 ; j < size ; j++ )
        spin[1][j] =  Heatbath(1, j, icall, iv) ;
}

/**************************** UPDATE: 1 ****************************/

void do_updatetot_MPI(int* icall, int* iv, int up_id, int dn_id, int myparity)
{
#pragma omp master
    if(myparity == 0)
    {
        MPI_Send(spin[1], size, MPI_INT, dn_id, 10, MPI_COMM_WORLD);
        MPI_Recv(spin[sizenp1], size, MPI_INT, up_id, 10, MPI_COMM_WORLD, &status);
    }
    else if (myparity == 1)
    {
        MPI_Recv(spin[sizenp1], size, MPI_INT, up_id, 10, MPI_COMM_WORLD, &status);
        MPI_Send(spin[1], size, MPI_INT, dn_id, 10, MPI_COMM_WORLD);
    }
    else if (myparity == 3)
        for (int j = 0; j < size; j++)
            spin[sizenp1][j] = spin[1][j];
#pragma omp barrier

    do_update_bulk_MPI(icall, iv);

#pragma omp master
    if(myparity == 0)
    {
        MPI_Send(spin[sizenp], size, MPI_INT, up_id, 11, MPI_COMM_WORLD);
        MPI_Recv(spin[0], size, MPI_INT, dn_id, 11, MPI_COMM_WORLD, &status);
    }
    else if (myparity == 1)
    {
        MPI_Recv(spin[0], size, MPI_INT, dn_id, 11, MPI_COMM_WORLD, &status);
        MPI_Send(spin[sizenp], size, MPI_INT, up_id, 11, MPI_COMM_WORLD);
    }
    else if (myparity == 3)
        for (int j = 0; j < size; j++)
            spin[0][j] = spin[sizenp][j];
#pragma omp barrier

#pragma omp master
    do_update_row1_MPI(icall, iv);
#pragma omp barrier
}

/**************************** HEATBATH *****************************/

int Heatbath(int i, int j, int* icall, int* iv)
{
    int trialspin;
    int k = staple(i, j);
    int m = k / 2 + 2;
    double rran = rndnum(icall, iv);

    if (rran < weight[m])
        trialspin = -1 ;
    else
        trialspin = 1 ;

    return trialspin;
}

/***************************** RNDNUM ******************************/

double rndnum(int* icall_o, int* iv)
{
    int ivn;
    double frndnum;
    int ir = 24;
    int is = 10;
    int irs = ir - is;
    int ir1 = ir - 1;
    int ikeep = ir;
    int ithrow = 24;
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

/*************************** READ INPUT ****************************/

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

/**************************** PRINT LOG ****************************/

void print_log()
{
    std::cout << "Ising model on a 2D square lattice" << std::endl ;
    std::cout << "Size of the lattice: " << size << std::endl ;
    std::cout << "Beta: " << beta << std::endl;
    std::cout << "Thermalisation sweeps: " << thermalisation << std::endl;
    std::cout << "Number of measurements: " << measurements << std::endl;

    if (istart == 1)
        std::cout << "Start: hot" << std::endl;
    else
        std::cout << "Start: cold" << std::endl;
}

/***************************** OUTPUT ******************************/

void write_measures_MPI(double elapsed)
{
    //for (int i = 0 ; i < measurements ; i++)
    //    outfile << inv_beta << "\t" << enetot[i] << "\t" << enetot[i] * enetot[i] << "\t" << magntot[i] << "\t" << magntot[i] * magntot[i] << "\n" ;
    outfile << size << "\t" << elapsed << "\n" ;
}

/*************************** INIT WEIGHT ***************************/

void initweight()
{
    for (int i = 0; i < 5; i++)
    {
        int j = 2 * (i - 2);
        weight[i] = 1.0 / (1.0 + exp(2 * j * beta)) ;
    }
}

/***************************** STAPLE ******************************/

int staple (int i, int j)
{
    int ifor = (i + 1) % size;
    int jfor = (j + 1) % size;
    int iback = ((i - 1) + size) % size;
    int jback = ((j - 1) + size) % size;

    return spin[ifor][j] + spin[iback][j] + spin[i][jfor] + spin[i][jback];
}
