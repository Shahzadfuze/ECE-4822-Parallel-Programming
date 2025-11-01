// file: p02.cc
//
// Autocorrelation computation in C++ (ECE 4822 Homework 1)
//

#include <cstdio>
#include <cstdlib>
#include <ctime>

// ---------------------------------------------------
// Generate N random values for x[n] in the range [-1,1]
// ---------------------------------------------------
void genRandomSignal(float* x, long N) {
  for (long i = 0; i < N; i++) {
    // Scale rand() [0,1] -> [-1,1]
    x[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
  }
}

// ---------------------------------------------------
// Compute autocorrelation function
// R[k] = sum_{n=0}^{N-1-k} x[n] * x[n+k],   k=0..K-1
// ---------------------------------------------------
bool autocor(float* R, float* x, long N, long K) {
  if (!R || !x) {
    return false;
  }

  for (long k = 0; k < K; k++) {
    float sum = 0.0f;
    for (long n = 0; n < N - k; n++) {
      sum += x[n] * x[n + k];
    }
    R[k] = sum;
  }

  return true;
}

// ---------------------------------------------------
// Main program
// Usage: p02.exe N K niter
// ---------------------------------------------------
int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stdout, "Usage: p02.exe N K niter\n");
    return 1;
  }

  long N = atoi(argv[1]);
  long K = atoi(argv[2]);
  long niter = atoi(argv[3]);

  // Allocate memory for signal and autocorrelation
  float* x = (float*)malloc(N * sizeof(float));
  float* R = (float*)malloc(K * sizeof(float));

  if (!x || !R) {
    fprintf(stderr, "Memory allocation failed\n");
    return 1;
  }

  srand((unsigned int)time(NULL)); // seed random generator

  // Loop through niter times
  for (long iter = 0; iter < niter; iter++) {
    genRandomSignal(x, N);

    if (!autocor(R, x, N, K)) {
      fprintf(stderr, "Autocorrelation failed\n");
      break;
    }

    // Display autocorrelation values
    //    fprintf(stdout, "Iteration %ld:\n", iter + 1);
    for (long k = 0; k < K; k++) {
      //      fprintf(stdout, "R[%ld] = %f\n", k, R[k]);
    }
  }

  free(x);
  free(R);

  return 0;
}
