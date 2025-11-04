#include "x.h"

// Goal of this program.
// Modify our original pointer matrix multiplication (hw_03) to run using N threads
// We can use "omp_set_num_theads(N)" function 


int main(int argc, char** argv){



  if(argc != 5){

    printf("Please use this format: \n <%s> <nrows> <ncols> <niter> <nthreads> \n", argv[0]); 
    
    return 1;
  }

  long nrows = atoi(argv[1]);
  long ncols = atoi(argv[2]);
  int niter = atoi(argv[3]);
  int nthreads = atoi(argv[4]);
  
  srand(time(NULL));


  // allocate rows
  float* mat1 = (float*)malloc(nrows * ncols * sizeof(float));
  float* mat2 = (float*)malloc(nrows * ncols * sizeof(float));
  float* mat3 = (float*)malloc(nrows * ncols * sizeof(float));

  randomizeMatrix(mat1, nrows, ncols);
  randomizeMatrix(mat2, nrows, ncols);


  //printMatrix(mat1, nrows, ncols);
  //    printMatrix(mat2, nrows, ncols);
  


  for(int i = 0; i < niter; i++){


    multiMatrix(mat3, mat2, mat1, nrows, ncols, nthreads);

  }

  //    printMatrix(mat3, nrows, ncols);
  // free memory

  free(mat1);
  free(mat2);
  free(mat3);


  
return 0;

}
