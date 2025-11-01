#include <Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace Eigen;

// Fill matrix with random integers 0–99
void randomizeMatrix(MatrixXd& mat) {
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      mat(i, j) = std::rand() % 100;
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <nrows> <ncols> <N>\n";
    return EXIT_FAILURE;
  }

  int nrows = std::atoi(argv[1]);
  int ncols = std::atoi(argv[2]);
  int N     = std::atoi(argv[3]);

  std::srand(static_cast<unsigned>(std::time(nullptr)));

  // A is (nrows x ncols), B is (ncols x nrows), result C is (nrows x nrows)
  MatrixXd A(nrows, ncols);
  MatrixXd B(ncols, nrows);
  MatrixXd C(nrows, nrows);

  // Measure runtime
  auto start = std::chrono::high_resolution_clock::now();

  randomizeMatrix(A);
  randomizeMatrix(B);


  
  for (int d = 0; d < N; ++d) {
    C.noalias() = A * B;  // optimized multiplication, avoids temporaries
  }

  /* std::cout << "Completed " << N << " multiplications of "
	    << nrows << "x" << ncols << " matrices in "
	    << elapsed << " seconds.\n";
  */
  return EXIT_SUCCESS;
}
