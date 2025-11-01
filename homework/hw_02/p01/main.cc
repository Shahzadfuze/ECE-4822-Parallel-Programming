// file: .
//

// local include files
//
#include "example.h"

// function: main
//
// Program Function:
//
//


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

  Returns true if multiplation was complete
  False if not 
*/
bool multiMatrix(float* mat3, float* mat2, float* mat1, long nrows, long ncols){

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

/*
  Main Function (Performs our matrix multiplation and send it to stdout)

  @param argc: Number of arguments 
  @param argv: List of Arguements

  Returns 0 if program ran sucessfully
  1 if Failed 
 */
int main(int argc, const char** argv) {

  if(argc < 4){
    fprintf(stdout, "please follow the format [p01.exe nrows ncols niter] \n nrows: the number of rows in the first matrix (and the number of columns in the second matrix) \n");
	    fprintf(stdout, "ncols: the number of columns in the first matrix (and the number of rows in the secix)\n");
	    fprintf(stdout, "niter: the number of iterations\n");
    return 1;
  }

  long nrows = atoi(argv[1]);
  long ncols = atoi(argv[2]);
  int niter = atoi(argv[3]);

  srand(time(NULL));


  // allocate rows
  float* mat1 = (float*)malloc(nrows * ncols * sizeof(float));
  float* mat2 = (float*)malloc(nrows * ncols * sizeof(float));
  float* mat3 = (float*)malloc(nrows * ncols * sizeof(float));

      randomizeMatrix(mat1, nrows, ncols);
    randomizeMatrix(mat2, nrows, ncols);

  
  for(int i = 0; i < niter; i++){

    // printMatrix(mat1, nrows, ncols);
    //printMatrix(mat2, nrows, ncols);
    multiMatrix(mat3, mat2, mat1, nrows, ncols);
    //printMatrix(mat3, nrows, ncols);
  }


  // free memory

  free(mat1);
  free(mat2);
  free(mat3);

  //Return Gracefully
  //
  return 0;


}
  
