#include "header.cuh"



int main(int argc, char** argv){

  
  if(argc != 4){
    printf("Please use this format: \n <%s> <nblocks> <nthreads> <nGPUs> \n", argv[0]); 
    return 1;
  }


    // Getting the number of GPUs on the node
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);


  long nrows =          10000;
  long ncols =          10000; 
  int niter =               100;
  int nblocks =  atoi(argv[1]);
  int nthreads = atoi(argv[2]);
  printf("%d\n",     nthreads);
  int gpuNum =   atof(argv[3]);

  //Define matrices for cuda and host


  float* A, *B, *C;
  long size = nrows * ncols * sizeof(float);
  
  // allocate rows
  A = (float*)malloc(size);
  B = (float*)malloc(size);
  C = (float*)malloc(size);


  
  int gpusToUse = std::min(deviceCount, gpuNum);
  int rowsPerGPU = nrows / gpusToUse;
  printf("Detected %d GPU(s), using %d\n", deviceCount, gpusToUse);

  
  randomizeMatrix(A, nrows, ncols);
  randomizeMatrix(A, nrows, ncols);


    for (int iter = 0; iter < niter; iter++) {
    // printf("\n--- Iteration %d ---\n", iter + 1);
    for (int g = 0; g < gpusToUse; g++) {

      cudaSetDevice(g);
      int startRow = g * rowsPerGPU;
      
      int nrowsA = (g == gpusToUse - 1) ? (nrows - startRow) : rowsPerGPU;
      
      size_t bytesA = nrowsA * ncols * sizeof(float);
      size_t bytesB = nrows * ncols * sizeof(float);
      size_t bytesC = nrowsA * ncols * sizeof(float);
      float *d_A, *d_B, *d_C;

      cudaMalloc(&d_A, bytesA);
      cudaMalloc(&d_B, bytesB);
      cudaMalloc(&d_C, bytesC);

      cudaMemcpy(d_A, A + startRow * ncols, bytesA, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, B, bytesB, cudaMemcpyHostToDevice);
      cudaMemset(d_C, 0, bytesC); // clear output for each iteration

      dim3 block(16, 16);
      dim3 grid((ncols + block.x - 1) / block.x, (nrowsA + block.y - 1) / block.y);

      // Time per GPU kernel
      cudaEvent_t iterStart, iterStop;

      cudaEventCreate(&iterStart);
      cudaEventCreate(&iterStop);
      cudaEventRecord(iterStart);


      gpu_mmulti<<<grid, block>>>(d_C, d_A, d_B, nrowsA, ncols);

      cudaDeviceSynchronize();
      cudaEventRecord(iterStop);
      cudaEventSynchronize(iterStop);

      float iter_ms = 0;
      cudaEventElapsedTime(&iter_ms, iterStart, iterStop);
      //printf("GPU %d iteration time: %.5f ms\n", g, iter_ms);

      
      cudaMemcpy(C + startRow * ncols, d_C, bytesC, cudaMemcpyDeviceToHost);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
      cudaEventDestroy(iterStart);
      cudaEventDestroy(iterStop);
    }
    cudaDeviceSynchronize();


  }


  

  
  

  free(A); free(B); free(C);
  
  
  return 0;

}

