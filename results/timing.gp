# gnuplot timing.gp
# ps2pdf timing.eps

set ylabel "Time To Run (seconds)"
set xlabel "Parallel units (threads, ranks, warps, etc)\n(MPI+OMP uses fixed 32 OMP threads)"
set title "Geometric Brownian Motion Strong Tests\nN=2048"
set grid ytics
set grid xtics
set yrange [0:0.05]
set term postscript color
set output "timing.eps"
plot \
     "mpi-omp-strong.csv" using 2:3 t "GBM with MPI+OMP" with linespoint,\
     "mpi-strong.csv" using 2:3 t "GBM with MPI" with linespoint,\
     "omp-strong.csv" using 2:3 t "GBM with OMP" with linespoint,\
     "cuda-strong.csv" using 3:4 t "GBM with CUDA" with linespoint,\
    # "mpi-omp-weak.csv" using 1:3 t "GBM with MPI+OMP" with linespoint,\
    # "mpi-weak.csv" using 1:3 t "GBM with MPI" with linespoint,\
    # "omp-weak.csv" using 1:3 t "GBM with OMP" with linespoint,\
    # "cuda-weak.csv" using 1:4 t "GBM with CUDA" with linespoint,\
    # "heat1d_blocking_weak_scales.csv" using 4:5 t "Heat1D Blocking (weak scale)" with linespoint,\
    # "heat1d_nonblocking_weak_scales.csv" using 4:5 t "Heat1D Nonblocking (weak scale)" with linespoint,\
    # "heat1d_sendrecv_weak_scales.csv" using 4:5 t "Heat1D Sendrecv (weak scale)" with linespoint,\
    # "heat1d_blocking_strong_scales.csv" using 4:5 t "Heat1D Blocking (strong scale)" with linespoint,\
    # "heat1d_blocking_weak_scales.csv" using 4:5 t "Heat1D Blocking (weak scale)" with linespoint,\
    # "heat1d_nonblocking_strong_scales.csv" using 4:5 t "Heat1D Nonblocking (strong scale)" with linespoint,\
    # "heat1d_sendrecv_strong_scales.csv" using 4:5 t "Heat1D Sendrecv (strong scale)" with linespoint,\
     
     
     

