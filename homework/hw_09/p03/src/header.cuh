// file: example_vector_add/header.cuh
// author: Leo Battalora (leo6@temple.edu)
//

// make sure definitions are only made once
//
#ifndef HEADER_CUH
#define HEADER_CUH

// include files
//
#include <stdio.h>
#include <omp.h>

// define constants
//


// function definitions
//
void randomizeMatrix(float* matrix, long nrows, long ncols);
void printMatrix(float* matrix, long nrows,long ncols);
__global__ void gpu_mmulti(float* C, const float* A, const float* B, int nrowsA, int ncols);
bool multiMatrix(float* mat3, float* mat2, float* mat1, long nrows, long ncols, int threads);


// kernel definitions (run on the GPU)
//

// end of include file
//
#endif
