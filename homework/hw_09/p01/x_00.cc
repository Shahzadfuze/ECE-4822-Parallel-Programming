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

  // Old Slow Matrix multiplication 
   /*
  if(!mat1 || !mat2 || !mat3){
    return false;
  }
  for (long i = 0; i < nrows; i++) {
    for (long k = 0; k < ncols; k++) {
      float a = mat1[i * ncols + k];
      for (long j = 0; j < nrows; j++) {
	mat3[i * nrows + j] += a * mat2[k * nrows + j];
      }
    }
  }
  */
#pragma omp parallel for num_threads(threads)
  for(long i = 0; i < nrows; i++){
    /*
      Setting the pointers to the rows in our result matrix
      as well as our first matrix 
     */
    float* c_row = mat3 + i * nrows;  
    float* a_row = mat1 + i * ncols;


    for(long k =0; k < ncols; k++){
      /*
	A is going to equal "A[i, k]"
	Well we are setting b_rows to be the rows for our other matrix mat2
       */
      float a = a_row[k];
      float* b_row = mat2 + k * nrows;
      


      float* c_ptr = c_row;
      float* b_ptr = b_row;
      
      /*
	This is the matrix multiplication, we are making the
	c_row = to a_row * b_row with pointers and we advance c and b 
       */
      for(long j = 0; j < nrows; j++){
	*c_ptr += a * (*b_ptr);
	c_ptr++;
	b_ptr++;

      }
    }
  }

  

  return true;
}
