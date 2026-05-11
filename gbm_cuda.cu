#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h> // measuring time in the absence of mpi timers
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h> // need to use CUDA's rng functions
#include <curand_kernel.h> // also necessary, otherwise nvcc can't find identifiers

// module load cuda
// nvcc -o gbm_cuda gbm_cuda.cu -lm -lcurand
// -lcurand is necessary so that nvcc links the curand rng libraries

// this file contains code to run a Geometric Brownian Motion (GBM) task parallelized using CUDA.
// See here for a basic overview of what GBM is:
// https://www.quantstart.com/articles/Geometric-Brownian-Motion/
// See here for a more in-depth explanation of how it works:
// https://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

// If a function does not explicitly identify a source where it was taken from,
// you may assume that I wrote it myself.

// error checking macro, taken from an assignment.
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

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
__device__ void print_gbm_matrix(double* gbm_matrix, int runs, int steps) {
  printf("DEBUG PRINTING GBM MATRIX\n");
  for (int i = 0; i < runs; i++) {
    for (int j = 0; j < steps; j++) {
      int current_address = i*steps+j;
      printf("%f ", gbm_matrix[current_address]);
    }
    printf("\n");
  }
}

// calculate the S_i value given S_0: S_i = S0 * exp(X(t))
// X(t) = (mu - 0.5*sigma**2)*t + sigma*sqrt(t)*normal_dist_randn(0, 1)
// t is a timedelta, difference in time between 0 and i
// the normal distribution rand simulates a random walk, while sqrt(t) scales it relative to the time difference

/**
 * @brief This function simulates a number of Geometric Brownian Motion trials.
 * 
 * It creates a 1d array interpreted as a 2d row-major matrix to store all values.
 * Each row then gets populated with intermediate values for a GBM trial.
 * This is a CUDA implementation, hence the __global__ tag.
 * 
 * @param gbm_matrix a CUDA device-side matrix (represented as a 1d array) to store all data in.
 * @param vars a gbm_variables struct containing all relevant variables. See the gbm_variables struct for details.
 * @param my_curandstate a curandState object, needed for CUDA random number generators.
 * 
 * @return Nothing. The final matrix stays in device memory awaiting cudamemcpy.
 */
__global__ void do_gbm_cuda(double *gbm_matrix, struct gbm_variables vars, curandState *my_curandstate) {
  // thread index; determines which rows in gbm_matrix each row should handle.
  int thread_idx = threadIdx.x;
  // each thread is assigned a number of GBM runs to calculate
  int start = (vars.runs / blockDim.x) * thread_idx;
  int end = (vars.runs / blockDim.x) * (thread_idx+1);
  // matrix of dims runs * steps
  // Each row is a GBM run spanning a number of steps
  //double* gbm_matrix = malloc(vars.runs*vars.steps*sizeof(double));

  // Now do the GBM
  // For each run:
  for (int i = start; i < end; i++) {
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
      // X(t) is a Wiener process and so by definition X(0) = 0 almost surely

      // calculate left and right hand side of what's inside the exp() func
      double left = vars.mu - 0.5*pow(vars.sigma, 2)*dt;
      //double right = sigma * sqrt(dt) * (double)normal_dist_randn(0, dt);
      // See comment after the function on a discussion on sqrt(dt).
      double right = vars.sigma * sqrt(dt) * (double)curand_normal(my_curandstate);
      gbm_matrix[current_address] = vars.s0 * exp(left+right);
    }
  }
  // if (vars.steps <= 12)
  //   print_gbm_matrix(gbm_matrix, vars.runs, vars.steps);
  //free(gbm_matrix);
}

/*
The random numbers need to be from a normal distribution with std = sqrt(dt).
Since curand_normal is locked to std=1, multiply its output by sqrt(dt) to scale its std.
*/

// Sets up a cuRAND kernel.
// Taken from here: https://stackoverflow.com/questions/18501081/generating-random-number-within-cuda-kernel-in-a-varying-range
__global__ void setup_kernel(curandState *state){
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(69420, idx, 0, &state[idx]);
  return;
}

// a partial eval for debug uses, and to conserve Bridges2 GPU compute units.
// I will execute full_run after all debugging.
int part_run() {
  double* gbm_matrix;

  struct gbm_variables vars; // host side
  //struct gbm_variables *dev_vars; // device side
  vars.mu = 0.10;
  vars.sigma = 0.20;
  vars.runs = 10;
  vars.steps = 10;
  vars.s0 = 10.0;
  // vars does not change size between updates so we allocate outside
  //cudaMalloc(&dev_vars, sizeof(vars));
  // set up a CUDA RNG state. It will be passed to the GBM kernel func later.
  curandState *d_state;
  cudaMalloc(&d_state, sizeof(curandState));
  int runs_steps[] = {64, 128, 256, 512, 1024, 2048};
  int cuda_thread_counts[] = {32, 64, 128, 256, 512, 1024};
  struct timeval stop, start;
  
  vars.runs = runs_steps[0];
  vars.steps = runs_steps[0];

  int blocks = 2; 
  int threads = 1024;
  
  // setup time trackers
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  // malloc gbm_matrix; its size changes between runs so we need to dynamically malloc it per run
  cudaMalloc(&gbm_matrix, vars.runs*vars.steps*sizeof(double));
  // error check calls taken from the CUDA MatMul with shared memory assignment.
  cudaCheckErrors("cudaMalloc failure");
  //cudaMemCpy(dev_vars, &vars, sizeof(vars), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemCpy H2D failure");
  
  setup_kernel<<<blocks, threads>>>(d_state);
  // do a weak scale test: threads scale with problem size
  cudaEventRecord(begin, 0);
  // format is <<<NUM_BLOCKS, BLOCK_WIDTH>>>
  do_gbm_cuda<<<blocks, threads>>>(gbm_matrix, vars, d_state);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);// make sure the end record doesn't run until gbm is all done
  cudaCheckErrors("GBM cuda execution failure");
  cudaFree(gbm_matrix);

  // do a strong scale test: set problem size of 2048
  cudaEventRecord(begin, 0);
  // format is <<<NUM_BLOCKS, BLOCK_WIDTH>>>
  do_gbm_cuda<<<blocks, threads>>>(gbm_matrix, vars, d_state);
  cudaEventRecord(end, 0);
  cudaEventSynchronize(end);// make sure the end record doesn't run until gbm is all done
  cudaCheckErrors("GBM cuda execution failure");
  cudaFree(gbm_matrix);

  float time;
  cudaEventElapsedTime(&time, begin, end);
  // time is measured in ms; divide by 1000 to get seconds
  time /= 1000;
  printf("size: %d, threads: %d, time (s): %f\n", runs_steps[0], threads, time);

  // If this is production code, you should run a cudaMemcpy() with cudaMemcpyDeviceToHost
  // to grab the GBM runs. Since this is for parallelism evaluation purposes only I will free the array.
  cudaFree(gbm_matrix);
  //cudaFree(dev_vars);
  cudaFree(d_state);
  return 0;
}


// a full eval of GBM parallelization in CUDA.
int full_run() {
  double* gbm_matrix;
  //double* gpu_gbm_matrix;
  //gbm_matrix = malloc(vars.runs*vars.steps*sizeof(double));
  // Set random seed to current computer time.
  // Not necessary for testing parallelism efficiency only. Useful if you want to use this to actually do work.
  //srand(time(NULL)); 
  struct gbm_variables vars;
  //struct gbm_variables *dev_vars;
  vars.mu = 0.10;
  vars.sigma = 0.20;
  vars.runs = 10;
  vars.steps = 10;
  vars.s0 = 10.0;
  // vars does not change size between updates so we allocate outside
  //cudaMalloc(&dev_vars, sizeof(vars));
  // set up a CUDA RNG state. It will be passed to the GBM kernel func later.
  curandState *d_state;
  cudaMalloc(&d_state, sizeof(curandState));
  setup_kernel<<<1,1>>>(d_state);
  int runs_steps[] = {64, 128, 256, 512, 1024, 2048};
  int cuda_thread_counts[] = {32, 64, 128, 256, 512, 1024};
  struct timeval stop, start;
  for (int i = 0; i < sizeof(runs_steps)/sizeof(int); i++) {
    
    vars.runs = runs_steps[i];
    vars.steps = runs_steps[i];
    // For simplicity's sake: 1 block
    int blocks = 1; 
    int threads = cuda_thread_counts[i];

    // setup time trackers
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    // malloc gbm_matrix; its size changes between runs so we need to dynamically malloc it per run
    cudaMalloc(&gbm_matrix, vars.runs*vars.steps*sizeof(double));
    // error check calls taken from the CUDA MatMul with shared memory assignment.
    cudaCheckErrors("cudaMalloc failure (weak)");
    //cudaMemcpy(dev_vars, &vars, sizeof(vars), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemCpy H2D failure");
    
    // do a weak scale test: threads scale with problem size
    cudaEventRecord(begin, 0);
    // format is <<<NUM_BLOCKS, BLOCK_WIDTH>>>
    do_gbm_cuda<<<blocks, threads>>>(gbm_matrix, vars, d_state);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);// make sure the end record doesn't run until gbm is all done
    cudaCheckErrors("GBM cuda execution failure");

    float time;
    cudaEventElapsedTime(&time, begin, end);
    time /= 1000;
    printf("size: %d, threads: %d, time (s): %f\n", runs_steps[i], threads, time);

    // do a strong scale test: fixed problem size of 2048
    // update dev_vars with new info
    vars.runs = 2048;
    vars.steps = 2048;
    //cudaMemcpy(dev_vars, &vars, sizeof(vars), cudaMemcpyHostToDevice);

    //update the gbm matrix; free and remalloc
    cudaFree(gbm_matrix);
    cudaMalloc(&gbm_matrix, vars.runs*vars.steps*sizeof(double));
    // error check calls taken from the CUDA MatMul with shared memory assignment.
    cudaCheckErrors("cudaMalloc failure (strong)");
    //cudaMemcpy(dev_vars, &vars, sizeof(vars), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemCpy H2D failure");

    cudaEventRecord(begin, 0);
    // format is <<<NUM_BLOCKS, BLOCK_WIDTH>>>
    do_gbm_cuda<<<blocks, threads>>>(gbm_matrix, vars, d_state);
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);// make sure the end record doesn't run until gbm is all done
    cudaCheckErrors("GBM cuda execution failure");

    //float time;
    cudaEventElapsedTime(&time, begin, end);
    time /= 1000;
    printf("size: %d, threads: %d, time (s): %f\n", vars.runs, threads, time);

    // If this is production code, you should run a cudaMemcpy() with cudaMemcpyDeviceToHost
    // to grab the GBM runs. Since this is for parallelism evaluation purposes only I will free the array.
    cudaFree(gbm_matrix);
  }

  //do_gbm(vars);
  // printf("%d\n",RAND_MAX);
  // for (int i = 0; i < 24; i++) {
  //   double myrand = normal_dist_randn(0,1);
  //   printf("%f\n", myrand);
  // }
  //cudaFree(dev_vars);
  cudaFree(d_state);
  return 0;
}

int main() {
  part_run();
  return 0;
}