#ifndef X_H
#define X_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdlib> 
#include <omp.h>
#include <vector>


void randomizeMatrix(float* matrix, long nrows, long ncols);

void printMatrix(float* matrix, long nrows,long ncols);

bool multiMatrix(float* mat3, float* mat2, float* mat1, long nrows, long ncols, int threads);


#endif // X_H
