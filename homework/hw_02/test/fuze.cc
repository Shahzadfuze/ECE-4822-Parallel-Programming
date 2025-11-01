#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define PI 3.141592653589793

// Compute unfactored autocorrelation
void autocorr_unfactored(double* x, int N, double* r, int p) {
  for (int k = 0; k <= p; k++) {
    r[k] = 0;
    for (int n = 0; n < N - k; n++)
      r[k] += x[n] * x[n + k];
  }
}

// Cholesky factorization of symmetric positive definite matrix
void cholesky(double** R, double** L, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0;
      for (int k = 0; k < j; k++)
	sum += L[i][k] * L[j][k];
      if (i == j)
	L[i][j] = sqrt(R[i][i] - sum);
      else
	L[i][j] = (R[i][j] - sum)/L[j][j];
    }
    for (int j = i+1; j < n; j++) L[i][j] = 0;
  }
}

int main() {
  int N = 100;            // number of samples
  double Ts = 0.01;       // sampling period
  double f0 = 1.0;        // sine frequency
  int p = 5;              // autocorr order

  // Allocate arrays
  double* x = (double*) malloc(N * sizeof(double));
  double* r = (double*) malloc((p+1) * sizeof(double));

  // Generate sine wave
  for (int n = 0; n < N; n++)
    x[n] = sin(2 * PI * f0 * n * Ts);

  // Unfactored autocorrelation
  autocorr_unfactored(x, N, r, p);

  printf("Unfactored autocorrelation:\n");
  for (int k = 0; k <= p; k++)
    printf("r[%d] = %f\n", k, r[k]);

  // Build Toeplitz autocorr matrix
  double** R = (double**) malloc((p+1) * sizeof(double*));
  for (int i = 0; i <= p; i++) {
    R[i] = (double*) malloc((p+1) * sizeof(double));
    for (int j = 0; j <= p; j++)
      R[i][j] = r[abs(i-j)];  // Toeplitz
  }

  // Allocate L
  double** L = (double**) malloc((p+1) * sizeof(double*));
  for (int i = 0; i <= p; i++)
    L[i] = (double*) calloc(p+1, sizeof(double));

  // Cholesky factorization
  cholesky(R, L, p+1);

  printf("\nFactored autocorrelation (Cholesky L):\n");
  for (int i = 0; i <= p; i++) {
    for (int j = 0; j <= p; j++)
      printf("%f ", L[i][j]);
    printf("\n");
  }

  // Free memory
  for (int i = 0; i <= p; i++) { free(R[i]); free(L[i]); }
  free(R); free(L); free(x); free(r);

  return 0;
}
