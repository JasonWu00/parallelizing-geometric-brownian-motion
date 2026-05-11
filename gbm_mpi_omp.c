#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // measuring time in the absence of mpi timers
#include <mpi.h>
#include <omp.h>

// mpicc -g -Wall -fopenmp -o gbm_mpi_omp gbm_mpi_omp.c -lm 

// this file contains code to run a Geometric Brownian Motion (GBM) task parallelized with OpenMPI.
// See here for a basic overview of what GBM is:
// https://www.quantstart.com/articles/Geometric-Brownian-Motion/
// See here for a more in-depth explanation of how it works:
// https://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

// If a function does not explicitly identify a source where it was taken from,
// you may assume that I wrote it myself.

// Using rand() instead of rand_r(&next) results in major efficiency loss.
// Remember to explain this in the report.
// Also explain why _Thread_local matters.
_Thread_local static unsigned int next = 1;

// Generates a random number in a gaussian (normal) distribution.
// This function is taken from here:
// https://github.com/rflynn/c/blob/master/rand-normal-distribution.c
static double gauss(){
  double x = (double)rand_r(&next) / RAND_MAX,
         y = (double)rand_r(&next) / RAND_MAX,
         z = sqrt(-2 * log(x)) * cos(2 * M_PI * y);
  return z;
}

// A struct to store all values required for a GBM run.
// It will be passed by value b/c it's not that big memory-wise.
struct gbm_variables {
  // drift
  double mu;
  // volatility
  double sigma;
  int runs;
  int steps;
  double s0;
};

// Prints a set of GBM runs from a given matrix pointer, with each row of numbers signifying 1 run.
// Not recommended for use with big GBM matrices due to command console size limitations.
void print_gbm_matrix(double* gbm_matrix, int runs, int steps) {
  printf("DEBUG PRINTING GBM MATRIX\n");
  for (int i = 0; i < runs; i++) {
    for (int j = 0; j < steps; j++) {
      int current_address = i*steps+j;
      printf("%f ", gbm_matrix[current_address]);
    }
    printf("\n");
  }
}

// Simulates a Brownian Motion step.
double brownian_step(double dt, double mu, double sigma) {
  // calculate left and right hand side of what's inside the exp() func
  double left = mu - 0.5*pow(sigma, 2)*dt;
  //double right = sigma * sqrt(dt) * (double)normal_dist_randn(0, dt);
  // See comment after the function on a discussion sqrt(dt).
  double right = sigma * sqrt(dt) * (double)gauss();
  return exp(left+right);
}

/*
The GBM function calls for W(t), where W is a Wiener function that simulates a random step.
This is the normal_dist_randn() function at work.
The requirements is that the std should == sqrt(dt).
Since gauss() has a std of 1, multiply by sqrt(dt) to adjust.
*/

// calculate the S_i value given S_0: S_i = S0 * exp(X(t))
// X(t) = (mu - 0.5*sigma**2)*t + sigma*sqrt(t)*normal_dist_randn(0, 1)
// t is a timedelta, difference in time between 0 and i
// the normal distribution rand simulates a random walk, while sqrt(t) scales it relative to the time difference

/**
 * @brief This function simulates a number of Geometric Brownian Motion trials.
 * 
 * It creates a 1d array interpreted as a 2d row-major matrix to store all values.
 * Each row then gets populated with intermediate values for a GBM trial.
 * This is a parallel implementation that uses OpenMPI to carry out work across many ranks.
 * 
 * @param vars a gbm_variables struct containing all relevant variables. See the gbm_variables struct for details.
 * 
 * @return Nothing. This function prints then frees the matrix.
 */
void do_gbm_mpi_omp(struct gbm_variables vars) {
  // matrix of dims runs * steps
  // Each row is a GBM run spanning a number of steps
  int myid, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);  /* get current process id */
  MPI_Comm_size(MPI_COMM_WORLD, &p);     /* get number of processes */
  int local_runs_count = vars.runs / p;
  double* gbm_matrix = malloc(local_runs_count*vars.steps*sizeof(double));

  // Now do the GBM
  // For each run:
  omp_set_num_threads(32); // using a fixed amount
  #pragma omp parallel shared(gbm_matrix,vars)
  {
    int my_thread_num = omp_get_thread_num();
    int threads_count = omp_get_num_threads();
    
    for (int i = 0; i < local_runs_count; i++) {
      // for each entry in a run:
      #pragma omp for //outer loop is already being parallelized by mpi
      for (int j = 0; j < vars.steps; j++) {
        int current_address = i*vars.steps+j; // each run has an offset of steps
        if (j == 0) { // every GBM run begins with s0
          gbm_matrix[current_address] = vars.s0;
          continue;
        }
        double dt = j * 1.0 / vars.steps; // treating a full run as 1 unit of time
        // pg 2 of the Columbia U pdf: S(t+h) = S(t) * e^(X(t+h)-X(t))
        // given that t = 0 (we are using S0 as the start point), S(h) = S0 * e^(X(h)-X(0))
        // X(0) should be 0; X(t) is a Wiener process and so by definition X(0) = 0 almost surely
        gbm_matrix[current_address] = vars.s0 * brownian_step(dt, vars.mu, vars.sigma);// - brownian_step(0, vars.mu, vars.sigma));
      }
    }
  }
  double* sum_gbm_matrix;
  if (myid == 0) {
    sum_gbm_matrix = malloc(vars.runs*vars.steps*sizeof(double));
  }
  // Gather the full GBM output to rank 0
  MPI_Gather(&gbm_matrix[0], local_runs_count*vars.steps, MPI_DOUBLE, &sum_gbm_matrix[0], local_runs_count*vars.steps, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  free(gbm_matrix);
  if (myid == 0) { // This is just for time evaluation purposes so I won't be returning the final results
    free(sum_gbm_matrix);
  }
}

int main() {
  // Set random seed to current computer time.
  // Not necessary for testing parallelism efficiency only. Useful if you want to use this to actually do work.
  //srand(time(NULL));
  MPI_Init(NULL,NULL);                 /* starts MPI */
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);  /* get current process id */
  int total_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);     /* get number of processes */
  //double start, end;
  struct gbm_variables vars;
  vars.mu = 0.10;
  vars.sigma = 0.20;
  vars.runs = 10;
  vars.steps = 10;
  vars.s0 = 10.0;
  int runs_steps[] = {32, 64, 128, 256, 512, 1024, 2048};
  int ranks_list[] = {1, 2, 4, 8, 16, 32, 64};
  double stop, start;

  int array_address = 0;
  for (int i = 0; i < sizeof(ranks_list)/sizeof(int); i++) {
    if (ranks_list[i] == total_ranks) {
      array_address = i;
    }
  }

  // do a weak scale: problem size scales with rank count
  vars.runs = runs_steps[array_address];
  vars.steps = runs_steps[array_address];
  start = MPI_Wtime();
  do_gbm_mpi_omp(vars);
  MPI_Barrier(MPI_COMM_WORLD);
  stop = MPI_Wtime();
  //double time_ms = ((stop.tv_sec - start.tv_sec) * 1000000.0 + stop.tv_usec - start.tv_usec) / 1000000.0;
  if (rank == 0)
    printf("size: %d, time (s): %f\n", runs_steps[array_address], stop-start);

  // do a strong scale: constant problem size of 2048
  vars.runs = 2048;
  vars.steps = 2048;
  start = MPI_Wtime();
  do_gbm_mpi_omp(vars);
  MPI_Barrier(MPI_COMM_WORLD);
  stop = MPI_Wtime();
  //double time_ms = ((stop.tv_sec - start.tv_sec) * 1000000.0 + stop.tv_usec - start.tv_usec) / 1000000.0;
  if (rank == 0)
    printf("size: %d, time (s): %f\n", vars.runs, stop-start);
  //do_gbm_mpi_omp(vars);
  // printf("%d\n",RAND_MAX);
  // for (int i = 0; i < 24; i++) {
  //   double myrand = normal_dist_randn(0,1);
  //   printf("%f\n", myrand);
  // }
  MPI_Finalize();
  return 0;
}
