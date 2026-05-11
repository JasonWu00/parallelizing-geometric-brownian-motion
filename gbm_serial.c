#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // measuring time in the absence of mpi timers

// gcc -g -Wall -o gbm_serial gbm_serial.c -lm

// this file contains code to run a Geometric Brownian Motion (GBM) task in serial.
// See here for a basic overview of what GBM is:
// https://www.quantstart.com/articles/Geometric-Brownian-Motion/
// See here for a more in-depth explanation of how it works:
// https://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

// If a function does not explicitly identify a source where it was taken from,
// you may assume that I wrote it myself.

// Generates a random number from a normal distribution given a mean and std.
// This function is taken from here:
// https://phoxis.org/2013/05/04/generating-random-numbers-from-normal-distribution-in-c/
// The parallelized codes use a different distribution function. This is due to some issues with
// static variables and thread safety. The serial code, being single-thread, need not worry.
double normal_dist_randn(double mu, double sigma){
  // the original implementation has uninitialized values which led to -nan returns
  // initing all vals with 0.0 fixed this
  double U1 = 0.0, U2 = 0.0, W = 0.0, mult = 0.0;
  static int call = 0;
  static double X1 = 0.0, X2 = 0.0;
 
  if (call == 1){
    call = !call;
    return (mu + sigma * (double) X2);
  }

  while (W >= 1 || W == 0){
    U1 = -1 + ((double) rand() * 1.0 / RAND_MAX) * 2;
    U2 = -1 + ((double) rand() * 1.0 / RAND_MAX) * 2;
    W = pow (U1, 2) + pow (U2, 2);
  }

  mult = sqrt ((-2 * log (W)) / W);
  X1 = U1 * mult;
  X2 = U2 * mult;

  call = !call;

  return (mu + sigma * (double) X1);
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
  double right = sigma * (double)normal_dist_randn(0, sqrt(dt));
  return exp(left+right);
}

/*
The GBM function calls for W(t), where W is a Wiener function that simulates a random step.
This is the normal_dist_randn() function at work.
The requirements is that the std should == sqrt(dt).
This can be accomplished using normal_dist_randn(0, sqrt(dt))
or sqrt(dt) * normal_dist_randn(0, 1)
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
 * This is a serial implementation; its structure will be used for parallelized versions in other files.
 * 
 * @param vars a gbm_variables struct containing all relevant variables. See the gbm_variables struct for details.
 * 
 * @return Nothing. This function prints then frees the matrix.
 */
void do_gbm(struct gbm_variables vars) {
  // matrix of dims runs * steps
  // Each row is a GBM run spanning a number of steps
  double* gbm_matrix = malloc(vars.runs*vars.steps*sizeof(double));

  // Now do the GBM
  // For each run:
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
  }
  if (vars.steps <= 12)
    print_gbm_matrix(gbm_matrix, vars.runs, vars.steps);
  free(gbm_matrix);
}

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
  struct timeval stop, start;
  for (int i = 0; i < sizeof(runs_steps)/sizeof(int); i++) {
    vars.runs = runs_steps[i];
    vars.steps = runs_steps[i];
    gettimeofday(&start, NULL);
    do_gbm(vars);
    gettimeofday(&stop, NULL);
    double time_ms = ((stop.tv_sec - start.tv_sec) * 1000000.0 + stop.tv_usec - start.tv_usec) / 1000000.0;
    printf("size: %d, time (s): %f\n", runs_steps[i], time_ms);
  }
  //do_gbm(vars);
  // printf("%d\n",RAND_MAX);
  // for (int i = 0; i < 24; i++) {
  //   double myrand = normal_dist_randn(0,1);
  //   printf("%f\n", myrand);
  // }
  return 0;
}
