#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // measuring time in the absence of mpi timers
#include <omp.h> // OMP parallelization

// gcc -g -Wall -fopenmp -o gbm_omp gbm_omp.c -lm

// this file contains code to run a Geometric Brownian Motion (GBM) task in parallel using OMP.
// See here for a basic overview of what GBM is:
// https://www.quantstart.com/articles/Geometric-Brownian-Motion/
// See here for a more in-depth explanation of how it works:
// https://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

// If a function does not explicitly identify a source where it was taken from,
// you may assume that I wrote it myself.

// Static number used for thread safe rand_r().

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
 * This is a parallel implementation using OMP.
 * 
 * @param vars a gbm_variables struct containing all relevant variables. See the gbm_variables struct for details.
 * @param threads number of OpenMP threads to use.
 * 
 * @return Nothing. This function prints then frees the matrix.
 */
void do_gbm_omp(struct gbm_variables vars, int threads) {
  // matrix of dims runs * steps
  // Each row is a GBM run spanning a number of steps
  double* gbm_matrix = malloc(vars.runs*vars.steps*sizeof(double));
  omp_set_num_threads(threads);
  #pragma omp parallel shared(gbm_matrix,vars,threads)
  {
    int my_thread_num = omp_get_thread_num();
    int threads_count = omp_get_num_threads();
    //printf("GBM running with %d threads\n", threads_count);
    // populate element 0 of each row with the initial value s0
    // not necessary any more; logic moved to main for loop
    // #pragma omp for
    // for (int i = 0; i < vars.runs*vars.steps; i+=vars.steps) {
    //   gbm_matrix[i] = vars.s0;
    // }
    // Now do the GBM
    // For each run:
    #pragma omp for
    for (int i = 0; i < vars.runs; i++) {
      // for each entry in a run:
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
      if (threads_count <= 8 && vars.steps <= 20) {
        printf("Thread %d worked on GBM trial %d\n", my_thread_num, i);
      }
    }
  } // parallel region ends here
  if (vars.steps <= 12)
    print_gbm_matrix(gbm_matrix, vars.runs, vars.steps);
  free(gbm_matrix);
}

/*
A given GBM run involves R runs each with S steps.
At each step the code calculates S(t) = S0 * e^(X(t))
where X(t) = (mu - 0.5*sigma**2)*t + sigma*normal_dist_randn(0, sqrt(t))
The normal distribution random number runs uses an inconsistent number of operations
due to the use of rand() in combination with the while-loop requirement

Outside of the random function, X(t) uses:
- 1 sqrt and 1 squaring
- 3 multiplications
- 2 add/subtract ops

The main GBM function then does 1 multiplication and 1 exp().

The loop in the integration approach for calculating pi makes n iterations.
Each itr calculates x and updates pi.
x involves a sum and a multiplication; 2 flop.
pi involves a sum and multiplication in the denominator, a multiplication in the numerator,
the division itself, and a sum with the old value of pi. 5 flop.
Memory assignments to update x and pi are not flop.
Each loop thus involves 7 floating point ops.
Total flops per loop = 7 * n 
divide by the duration for an average mflop/s value.
*/

int main() {
  // Set random seed to current computer time.
  // Not necessary for testing parallelism efficiency only. Useful if you want to use this to actually do work.
  //srand(time(NULL)); 
  struct gbm_variables vars;
  vars.mu = 0.10;
  vars.sigma = 0.20;
  vars.runs = 10;
  vars.steps = 10;
  vars.s0 = 10.0;
  int runs_steps[] = {32, 64, 128, 256, 512, 1024, 2048};
  int omp_thread_counts[] = {1, 2, 4, 8, 16, 32, 64};
  vars.runs = 16;
  vars.steps = 16;
  //do_gbm_omp(vars, 4);
  //printf("finished testing gbm run\n");
  //return 0;
  //struct timeval stop, start;
  for (int i = 0; i < sizeof(runs_steps)/sizeof(int); i++) {
    vars.runs = runs_steps[i];
    vars.steps = runs_steps[i];
    // do a weak scale test: threads scale with problem size
    double start = omp_get_wtime();
    do_gbm_omp(vars, omp_thread_counts[i]);
    double end = omp_get_wtime();
    //double time_ms = ((stop.tv_sec - start.tv_sec) * 1000000.0 + stop.tv_usec - start.tv_usec) / 1000.0;
    printf("size: %d, threads: %d, time (s): %f\n", runs_steps[i],  omp_thread_counts[i], end - start);
  }
  for (int i = 0; i < sizeof(runs_steps)/sizeof(int); i++) {
    // do a strong scale test: problem size is constant
    vars.runs = 2048;
    vars.steps = 2048;
    double start = omp_get_wtime();
    do_gbm_omp(vars, omp_thread_counts[i]);
    double end = omp_get_wtime();
    //double time_ms = ((stop.tv_sec - start.tv_sec) * 1000000.0 + stop.tv_usec - start.tv_usec) / 1000.0;
    printf("size: %d, threads: %d, time (s): %f\n", 2048,  omp_thread_counts[i], end - start);
  }
  //do_gbm(vars);
  // printf("%d\n",RAND_MAX);
  // for (int i = 0; i < 24; i++) {
  //   double myrand = normal_dist_randn(0,1);
  //   printf("%f\n", myrand);
  // }
  return 0;
}
