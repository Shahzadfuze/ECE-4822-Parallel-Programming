#include <stdio.h>
#include <stdlib.h>

// How to compile and run
// nvcc -allow-unsupported-compiler -O2 -o p01.exe p01.cu
//  nice -19 sbatch --partition=gpu run.sh
//

/*
  Multiplies our input matrices m1 and m2, and place the results into
  m3.
  This __global__ allows for us to run this function on the gpus 
  
  
 */
__global__ void gpu_mmulti(float* m3, float* m2, float* m1, long nrows, long ncols, long niter){  
  
  if(!m1 || !m2 || !m3){
    printf("You mutliplication doesn't work\n");
  }
  
  for (long i = 0; i < nrows; i++) {
    for (long j = 0; j < nrows; j++) {
      float sum = 0;
      for (long k = 0; k < ncols; k++) {
	sum += m1[i * ncols + k] * m2[k * ncols + j];
      }
      m3[i * ncols + j] = sum;
    }
  }

  
}



/*
  Populating the matrix with values between 0 and 10 before he do our calculations
 */

void randomizeMatrix(float* matrix, long nrows, long ncols){
  for(long i = 0; i < nrows; i++){
    for(long j = 0; j < ncols; j++){
      matrix[i * ncols + j] = (float)(rand() % 10);
    }
  }

}


void printMatrix(float* matrix, long nrows, long ncols){
  for(long i = 0; i < nrows; i++){
    for(long j = 0; j < ncols; j++){
      fprintf(stdout, "%6.6f ", matrix[i * ncols + j]); 
    }
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "\n");
}



int main(int argc, char** argv){
  
  if(argc != 4){
    fprintf(stdout, "Please run the program like\n <%s> <nrows> <ncols> <nitr>\n", argv[0]);
    return 1;
  }

  srand(time(NULL));
  // Reading the GPU time
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  
  
  long nrows = atof(argv[1]);
  long ncols = atof(argv[2]);
  long niter = atof(argv[3]);

  
  float *d_m1, *d_m2, *d_m3;
  
  // Allocate in CPU domain
  float* mat1 = (float*)malloc(nrows * ncols * sizeof(float));
  float* mat2 = (float*)malloc(nrows * ncols * sizeof(float));
  float* mat3 = (float*)malloc(nrows * ncols * sizeof(float));


    
  // Allocate in GPU domain
  cudaMalloc((void**)&d_m1, sizeof(float) * nrows * ncols);
  cudaMalloc((void**)&d_m2, sizeof(float) * nrows * ncols);
  cudaMalloc((void**)&d_m3, sizeof(float) * nrows * ncols);
  
 
  randomizeMatrix(mat1, nrows, ncols);
  randomizeMatrix(mat2, nrows, ncols);
  
  
  // Copy mat1 and mat2 to GPU
  //
  cudaMemcpy(d_m1, mat1, sizeof(float) * ncols * nrows, cudaMemcpyHostToDevice);
  cudaMemcpy(d_m2, mat2, sizeof(float) * ncols * nrows, cudaMemcpyHostToDevice);  

  cudaEventRecord(start);

  for(long i = 0; i < niter; i++){
    gpu_mmulti<<<1, 1>>>(d_m3, d_m2, d_m1, nrows, ncols, niter);
    cudaDeviceSynchronize();
  }
  

  
  cudaMemcpy(mat3, d_m3, sizeof(float) * nrows * ncols, cudaMemcpyDeviceToHost);


  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  /*
  printMatrix(mat1, nrows, ncols);
  printMatrix(mat2, nrows, ncols);
  printMatrix(mat3, nrows, ncols);
  */
  float ms = 0;

  cudaEventElapsedTime(&ms, start, stop);
  fprintf(stdout, "---------------\nGPU Time Execution\n---------------\n %.5f ms\n", ms);

  
  free(mat1);
  free(mat2);
  free(mat3);

  
  cudaFree(d_m1);
  cudaFree(d_m2);
  cudaFree(d_m3);
  
  return 1;
}
