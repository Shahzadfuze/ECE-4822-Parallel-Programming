#include "x.h"


/*
  Putting randomized numbers from 0-10 inside our matrix
  
  @param matrix: pointer to the matrix 
  @param nrows: number of given rows
  @param ncols: number of cols
*/
void randomizeMatrix(float* matrix, long nrows, long ncols){

  for(long i = 0; i < nrows; i++){
    for(long j = 0; j < ncols; j++){
      matrix[i * ncols + j] = (float)(rand() % 10);
    }
  }

}


/*
  Printing our matrix

  @param matrix: pointer to the matrix we want to print 
  @param nrows: rows of the matrix
  @param ncols: number o columns in our matrix 
*/
void printMatrix(float* matrix, long nrows,long ncols){
  for (long i = 0; i < nrows; i++) {
    for (long j = 0; j < ncols; j++) {
      printf("%6.6f ", matrix[i * ncols + j]);
    }
    printf("\n");
  }
  printf("\n");
}






/*
  Multiplies mat2 and mat1 to get mat3
  
  @param mat3: pointer to our resulting matrix
  @param mat2: pointer to our matrix 2 
  @param mat1: pointer to our matrix 1 
  @param nrows: number of rows in our matrix
  @param ncols: number of columns
  @prama nthreads: number of threads we want to use
  Returns true if multiplation was complete
  False if not 
*/

bool multiMatrix(float* mat3, float* mat2, float* mat1, long nrows, long ncols, int threads){

  omp_set_num_threads(threads);
  if(!mat1 || !mat2 || !mat3){
    return false;
  }

#pragma omp parallel for
  for (long i = 0; i < nrows; i++) {
    for (long j = 0; j < nrows; j++) {
      float sum = 0;
      for (long k = 0; k < ncols; k++) {
	sum += mat1[i * ncols + k] * mat2[k * nrows + j];
      }
      mat3[i * nrows + j] = sum;
    }
  }
  return true;
}
