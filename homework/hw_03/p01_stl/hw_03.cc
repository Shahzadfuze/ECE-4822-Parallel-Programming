#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

// Flattened 2D matrix: element(i,j) = mat[i * ncols + j]
using Matrix = std::vector<long>;

// Fill matrix with random values 0–99
void randomizeMatrix(Matrix& mat, long nrows, long ncols) {
  for (long i = 0; i < nrows * ncols; ++i) {
    mat[i] = static_cast<long>(std::rand() % 10);
  }
}

// Matrix multiplication: C = A * B
bool multiplyMatrix(const Matrix& A, const Matrix& B, Matrix& C,
		    long nrows, long ncols) {
  std::vector<double> B_T(nrows * ncols);  // Transposed B: shape (nrows x ncols)
  for (long i = 0; i < ncols; ++i) {
    for (long j = 0; j < nrows; ++j) {
      B_T[j * ncols + i] = B[i * nrows + j];
    }
  }

  // Step 2: Matrix multiplication using cache-friendly access
  for (long i = 0; i < nrows; ++i) {
    for (long j = 0; j < nrows; ++j) {
      double sum = 0.0;
      for (long k = 0; k < ncols; ++k) {
	// Now both A and B_T are accessed row-wise — better for cache
	sum += A[i * ncols + k] * B_T[j * ncols + k];
      }
      C[i * nrows + j] = sum;
    }
}
  return true;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <nrows> <ncols> <N>\n";
    return EXIT_FAILURE;
  }

  long nrows = std::atoi(argv[1]);
  long ncols = std::atoi(argv[2]);
  long N     = std::atoi(argv[3]);



  Matrix A(nrows * ncols);
  Matrix B(ncols * nrows);
  Matrix C(nrows * nrows);

  // Measure runtime




      randomizeMatrix(A, nrows, ncols);
    randomizeMatrix(B, ncols, nrows);

  
  for (long d = 0; d < N; ++d) {
    multiplyMatrix(A, B, C, nrows, ncols);
  }

  return EXIT_SUCCESS;
}
