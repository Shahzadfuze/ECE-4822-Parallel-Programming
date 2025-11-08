#include "header.cuh"



int main(int argc, char** argv){

  
  if(argc != 3){
    printf("Please use this format: \n <%s> <nblocks> <nthreads> \n", argv[0]); 
    return 1;
  }


  long nrows =          10000;
  long ncols =          10000; 
  int niter =             1000;
  int nblocks =  atoi(argv[1]);
  int nthreads = atoi(argv[2]);
  printf("%d\n",     nthreads);

  //Define matrices for cuda and host

  float* d_A, *d_B,* d_C;
  float* A, *B, *C;
  long size = nrows * ncols * sizeof(float);
  
  // allocate rows
  A = (float*)malloc(size);
  B = (float*)malloc(size);
  C = (float*)malloc(size);

  randomizeMatrix(A, nrows, ncols);
  randomizeMatrix(A, nrows, ncols);


  for (int iter = 0; iter < niter; iter++) {
  
    // Allocate for cuda
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy the mem from host to cuda
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
  

    dim3 blockDim(nthreads, nthreads);
    dim3 gridDim(nblocks, nblocks);



    // GPU timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  

    cudaEventRecord(start);
    gpu_mmulti<<<gridDim, blockDim>>>(d_C, d_A, d_B, nrows, ncols);  // launch kernel
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);


  }

    free(A); free(B); free(C);  
  
  return 0;

}

