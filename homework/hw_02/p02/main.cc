// file: p02.c
//
// local include files
//

// function: main
//
// Program Function:
//    ECE 4822 Homework 2 - Fast Algorithms in C
//    Computes the autocorrelation of a signal using FFT
//    Complexity: O(N log N)
//
// Usage:
//    ./p02 N K niter
//      N     = signal length
//      K     = number of lags
//      niter = number of iterations
//
// Revisions:
//    - HW01: Naive O(N^2) autocorrelation
//    - HW02: Cooley–Tukey FFT-based O(N log N) autocorrelation
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>


// 
// Complex struct
typedef struct {
  double re;
  double im;
} Complex;

// Complex helpers
Complex c_add(Complex a, Complex b) { return (Complex){a.re + b.re, a.im + b.im}; }
Complex c_sub(Complex a, Complex b) { return (Complex){a.re - b.re, a.im - b.im}; }
Complex c_mul(Complex a, Complex b) { return (Complex){a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re}; }
Complex c_conj(Complex a) { return (Complex){a.re, -a.im}; }

// Recursive FFT
void fft(Complex* a, int n, int invert) {
  if (n == 1) return;
  Complex *a0 = (Complex*) malloc(n/2 * sizeof(Complex));
  Complex *a1 = (Complex*) malloc(n/2 * sizeof(Complex));
  for (int i = 0; 2*i < n; i++) {
    a0[i] = a[2*i];
    a1[i] = a[2*i+1];
  }
  fft(a0, n/2, invert);
  fft(a1, n/2, invert);

  
  
  double ang = 2*M_PI/n * (invert ? -1 : 1);
  Complex w = {1,0}, wn = {cos(ang), sin(ang)};
  for (int i = 0; 2*i < n; i++) {
    Complex t = c_mul(w, a1[i]);
    a[i] = c_add(a0[i], t);
    a[i+n/2] = c_sub(a0[i], t);
    if (invert) {
      a[i].re /= 2; a[i].im /= 2;
      a[i+n/2].re /= 2; a[i+n/2].im /= 2;
    }
    w = c_mul(w, wn);
  }
  free(a0); free(a1);
}

// Random signal in [-1,1]
void genRandomSignal(Complex* x, long N) {
  for (long i = 0; i < N; i++) {
    double val = 2.0 * ((double)rand()/RAND_MAX) - 1.0;
    x[i].re = val; x[i].im = 0.0;
  }
}

// FFT-based autocorrelation (factored)
int autocor_fft(double* R, Complex* X, long N, long K, long size) {
  // Forward FFT
  fft(X, size, 0);

  // Multiply by conjugate
  for (long i = 0; i < size; i++)
    X[i] = c_mul(X[i], c_conj(X[i]));

  // Inverse FFT
  fft(X, size, 1);

  // Copy first K results
  for (long k = 0; k < K; k++)
    R[k] = X[k].re;

  return 1;
}

// ---------------------------------------------------
// Main
// ---------------------------------------------------
int main(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: %s N K niter\n", argv[0]);
    return 1;
  }

  long N = atol(argv[1]);
  long K = atol(argv[2]);
  long niter = atol(argv[3]);

  // Find FFT size (next power of 2 >= 2N)
  long size = 1;
  while (size < 2*N) size <<= 1;

  // Allocate once (factored)
  Complex* x = (Complex*) malloc(size * sizeof(Complex));
  double* R  = (double*)  malloc(K * sizeof(double));
  if (!x || !R) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  srand((unsigned int)time(NULL));

  for (long iter = 0; iter < niter; iter++) {
    // Generate signal
    genRandomSignal(x, N);
    for (long i = N; i < size; i++) { x[i].re = 0.0; x[i].im = 0.0; }

    // Compute autocorrelation
    if (!autocor_fft(R, x, N, K, size)) {
      fprintf(stderr, "Autocorrelation failed\n");
      break;
    }
    
    
    /*
      printf("Iteration %ld:\n", iter+1);
    for (long k = 0; k < K; k++)
      printf("R[%ld] = %f\n", k, R[k]);
    */
  }

  free(x);
  free(R);
  return 0;
}
