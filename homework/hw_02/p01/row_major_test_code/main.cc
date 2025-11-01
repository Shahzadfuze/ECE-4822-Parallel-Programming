// file: .
//

// local include files
//
#include "example.h"

// function: main
//
// Program Function:
// For me to understand how row-major code works 
//
//



/*
  Populate Matrix
  mat = matrix
  nrows = row size
  ncols = cols size

  Populate our matrices with any number
  
 */
void popMat(int* mat, int nrows, int ncols){
  for(int i = 0; i < nrows; i++){
    for(int j = 0; j < ncols; j++){


      /*
	i = 3
	ncols = 3
	j = 1

	mat[ 3 * 3 + 1] = mat[10]

	which is our index in the block of memory for

	mat[3][1] = mat[10] 
      */
      
      mat[i * ncols + j] = i; 
    }
  }
}




/*
  Prints the provided matrix
  mat = matrix
  nrows = size of row
  ncols = size of cols 
  
 */
void printMat(int* mat, int nrows, int ncols){
  for(int i = 0; i < nrows; i++){

    for(int j = 0; j < ncols; j++){
      fprintf(stdout, "%d ", mat[i * ncols + j]);
    }
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "\n");
}


/*
  Multiplies our Matrices

  mat1/mat2 = Our two given matrices
  mat3 = Output Matrix
  nrows = size of rows
  ncols = size of cols 
  
 */
void multiMat(int* mat1, int* mat2, int* mat3, int nrows, int ncols){
  
  /*
    This works but is slow
   */
  for(int i = 0; i < nrows; i++){
    for(int j = 0; j < ncols; j++){
      for(int k = 0; k < ncols; k++){
	mat3[i * ncols + j] += mat1[i * ncols + k] * mat2[k * ncols + j]; 
      }
    }
  }
}



void fastMultiMat(int* mat1, int* mat2, int* mat3, int nrows, int ncols){
  

  for(int i = 0; i < nrows; i++){

    int* c_ptr = mat3 + i + ncols; 
    int* a_ptr = mat1 + i + ncols;

    for(int j = 0; j < ncols; j++){
      int* b = mat2 + k + ncols;
      int a_n = a[k]; 
    }

    for(int j = 0; j < ncols; j++){
      c[j] += a_n * b[j];

    }
    
  }
  

  
}




int main(int argc, const char** argv) {

  // Make sure nrows and ncols are provided
  if (argc != 3){
    fprintf(stdout, "Error \n");
    return 1;
  }

  // Set nrows and ncols to given sizes 
  int nrows = atoi(argv[1]);
  int ncols = atoi(argv[2]);

  // Declaring our matrices with fixed given size

  /*
    Creates a big chunk of memory size of row*cols
    to store our matrices in 
   */
  
  int *mat1 = (int *)malloc(nrows*ncols*sizeof(int));
  int *mat2 = (int *)malloc(nrows*ncols*sizeof(int));
  int *mat3 = (int *)malloc(nrows*ncols*sizeof(int));

  
  // Filling our matrixing with a number (10 in this case)
  popMat(mat1, nrows, ncols);
  popMat(mat2, nrows, ncols);

  multiMat(mat1, mat2, mat3, nrows, ncols);
  
  printMat(mat1, nrows, ncols);
  printMat(mat2, nrows, ncols);
  printMat(mat3, nrows, ncols);

  free(mat1);
  free(mat2);
  free(mat3);
  return 0; 
}
