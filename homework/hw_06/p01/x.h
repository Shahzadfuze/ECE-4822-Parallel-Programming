#ifndef X_H
#define X_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdlib> 
#include <omp.h>
#include <vector>


typedef struct {
  float b[3];  // feed-forward coefficients: b0, b1, b2
  float a[3];  // feedback coefficients: a0=1, a1, a2
  float w1;    // state variable w[n-1]
  float w2;    // state variable w[n-2]
} Biquad;

float fir_filter(float input, float *coef, int n, float *history);
float fir_filter_circ(float input, const float* coef, int n, float* history, int* index);

#endif // X_H
