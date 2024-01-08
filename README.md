# GPU Programming

## Description:

This CUDA C program is designed for parallel computation on GPUs and appears to perform calculations related to astronomy or galaxy simulations. The primary goal seems to be computing the angular correlation function for real and simulated galaxies. The program uses three CUDA kernels (calculate_faster_angle_DR, calculate_faster_angle_DD, and calculate_faster_angle_RR) to efficiently handle the computation in parallel.

The code initializes and reads real and simulated galaxy data from input files, computes sine and cosine values for the angles on both the CPU and GPU, and then launches GPU kernels to calculate the angular correlation function. The results are copied back to the host, and the program prints relevant information, including the calculated angular correlation function, histograms, and performance metrics.

## Recommended background:

Programming experience in C/C++, although some have started with very little C knowledge.

## Installation

When Dione opens, load the modules
```shell
module load cuda
module load GCC
```

## Compile

You can test using the template of the project
```shell
srun -p gpu -n 1 -t 10:00 --mem=1G -e err.txt -o out.txt galaxy data_100k_arcmin.dat rand_100k_arcmin.dat omega.out
```