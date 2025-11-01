// ---------------------------------------------------
// Factored autocorrelation computation using Cholesky
// ---------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

// ---------------------------------------------------
// Generate sine wave signal
// x[n] = sin(2*pi*f0*n*Ts)
// ---------------------------------------------------
void genSineSignal(float* x, long N, float f0, float Ts) {
  for (long n = 0; n < N; n++) {
    x[n] = sinf(2.0f * 3.14159265f * f0 * n * Ts);
  }
}

// ---------------------------------------------------
// Compute unfactored autocorrelation R[k]
// R[k] = sum_{n=0}^{N-1-k} x[n]*x[n+k]
// ---------------------------------------------------
bool autocor(float* R, float* x, long N, long K) {
  if (!R || !x) return false;

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
// Build Toeplitz matrix from autocorrelation
// ---------------------------------------------------
void buildToeplitz(float** Rmat, float* R, long K) {
  for (long i = 0; i < K; i++) {
    for (long j = 0; j < K; j++) {
      long idx = abs(i - j);
      Rmat[i][j] = R[idx];
    }
  }
}

// ---------------------------------------------------
// Cholesky decomposition of symmetric positive-definite matrix
// Rmat = L * L^T
// ---------------------------------------------------
void cholesky(float** Rmat, float** L, long K) {
  for (long i = 0; i < K; i++) {
    for (long j = 0; j <= i; j++) {
      float sum = 0.0f;
      for (long k = 0; k < j; k++)
	sum += L[i][k] * L[j][k];

      if (i == j)
	L[i][j] = sqrtf(Rmat[i][i] - sum);
      else
	L[i][j] = (Rmat[i][j] - sum) / L[j][j];
    }
    for (long j = i + 1; j < K; j++)
      L[i][j] = 0.0f;
  }
}

// ---------------------------------------------------
// Main program
// ---------------------------------------------------
int main() {
  long N = 100;      // number of samples
  long K = 6;        // autocorr order
  float f0 = 1.0f;   // sine frequency
  float Ts = 0.01f;  // sampling period

  // Allocate memory
  float* x = (float*)malloc(N * sizeof(float));
  float* R = (float*)malloc(K * sizeof(float));
  float** Rmat = (float**)malloc(K * sizeof(float*));
  float** L = (float**)malloc(K * sizeof(float*));
  for (long i = 0; i < K; i++) {
    Rmat[i] = (float*)malloc(K * sizeof(float));
    L[i] = (float*)malloc(K * sizeof(float));
  }

  // Generate sine wave
  genSineSignal(x, N, f0, Ts);

  // Compute unfactored autocorrelation
  autocor(R, x, N, K);

  printf("Unfactored autocorrelation:\n");
  for (long k = 0; k < K; k++)
    printf("R[%ld] = %f\n", k, R[k]);

  // Build Toeplitz matrix
  buildToeplitz(Rmat, R, K);

  // Cholesky factorization
  cholesky(Rmat, L, K);

  printf("\nFactored autocorrelation (Cholesky L):\n");
  for (long i = 0; i < K; i++) {
    for (long j = 0; j < K; j++)
      printf("%f ", L[i][j]);
    printf("\n");
  }

  // Free memory
  free(x); free(R);
  for (long i = 0; i < K; i++) {
    free(Rmat[i]); free(L[i]);
  }
  free(Rmat); free(L);

  return 0;
}
