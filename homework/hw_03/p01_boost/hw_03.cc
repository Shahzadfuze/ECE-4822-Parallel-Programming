// file: hw_03/hw_03.cc
//

// local include files
//
#include <Cmdl.h>
#include <Fe.h>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <cstdlib>
#include <ctime>
#include <iostream>





namespace ublas = boost::numeric::ublas;


// main: ...
//
// This is a driver program...
//
int main(int argc, const char** argv) {

    if (argc != 4){
      std::cout << "./hw_03.exe <nrows> <ncols> <N>\n";
      return EXIT_FAILURE;
    }

    std::srand(std::time(nullptr));

    long nrows = std::atoi(argv[1]);
    long ncols = std::atoi(argv[2]);
    long N = std::atoi(argv[3]);

    // Use double for speed and column-major for better performance
    ublas::matrix<double, ublas::column_major> A(nrows, ncols);
    ublas::matrix<double, ublas::column_major> B(ncols, nrows); // Note: dimensions for multiplication
    ublas::matrix<double, ublas::column_major> C(nrows, nrows); // pre-allocate

    // Randomize A and B
    for(long i = 0; i < nrows; i++)
      for(long j = 0; j < ncols; j++)
	A(i,j) = std::rand() % 100;

    for(long i = 0; i < ncols; i++)
      for(long j = 0; j < nrows; j++)
	B(i,j) = std::rand() % 100;

    // Perform multiplication N times
    for(long d = 0; d < N; d++)
      ublas::noalias(C) = prod(A, B); // Avoid temporaries

    return EXIT_SUCCESS;
}
//
// end of file
