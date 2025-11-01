#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>



// Using mmulti and randomize from previous hw_05
__global__ void gpu_mmulti(float* C, const float* A, const float* B, int nrowsA, int ncols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nrowsA && col < ncols) {
    float sum = 0;
    for (int k = 0; k < ncols; ++k)
      sum += A[row * ncols + k] * B[k * ncols + col];
    C[row * ncols + col] = sum;
  }
}

void randomize(float* M, int rows, int cols) {
  for (int i = 0; i < rows * cols; ++i)
    M[i] = static_cast<float>(rand() % 10);
}


void printMatrix(const float* matrix, long nrows, long ncols, const char* name) {
  printf("\nMatrix %s (%ld x %ld):\n", name, nrows, ncols);
  for (long i = 0; i < nrows; i++) {
    for (long j = 0; j < ncols; j++) {
      printf("%8.2f ", matrix[i * ncols + j]);
    }
    printf("\n");
  }
  printf("\n");
}



void print_stats(int deviceCount){
  printf("Number of GPUs:  %d\n", deviceCount);
  for (int i = 0; i < deviceCount; i++) {    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    
    printf("  Device name: %s\n", prop.name);
    
    printf("  Memory Clock Rate (MHz): %d\n",
	   prop.memoryClockRate/1024);
    printf("  Memory Bus Width (bits): %d\n",
	   prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
	   2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Total global memory (Gbytes) %.1f\n",(float)(prop.totalGlobalMem)/1024.0/1024.0/1024.0);
    printf("  Shared memory per block (Kbytes) %.1f\n",(float)(prop.sharedMemPerBlock)/1024.0);
    printf("  minor-major: %d-%d\n", prop.minor, prop.major);
    printf("  Warp-size: %d\n", prop.warpSize);
    printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
    printf("  Concurrent computation/communication: %s\n\n",prop.deviceOverlap ? "yes" : "no");
  }

}


int main(int argc, char** argv){

  // Getting the number of GPUs on the node
  int deviceCount = 0;
  cudaGetDeviceCount(&deviceCount);

  float total_ms = 0.0f;
  if(argc != 5){
    fprintf(stdout, "Please run the program like\n <%s> <nrows> <ncols> <nitr> <gpuNumber>\n", argv[0]);
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
  int gpuNum = atof(argv[4]);

  
  // Host allocation
  float* h_A = (float*)malloc(nrows * ncols * sizeof(float));
  float* h_B = (float*)malloc(nrows * ncols * sizeof(float));
  float* h_C = (float*)malloc(nrows * ncols * sizeof(float));

  randomize(h_A, nrows, ncols);
  randomize(h_B, nrows, ncols);

  int gpusToUse = std::min(deviceCount, gpuNum);
  int rowsPerGPU = nrows / gpusToUse;
  printf("Detected %d GPU(s), using %d\n", deviceCount, gpusToUse);
  


  cudaEventRecord(start);


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

      cudaMemcpy(d_A, h_A + startRow * ncols, bytesA, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice);
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

      total_ms += iter_ms;
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&iter_ms, start, stop);
      
      cudaMemcpy(h_C + startRow * ncols, d_C, bytesC, cudaMemcpyDeviceToHost);
      cudaFree(d_A);
      cudaFree(d_B);
      cudaFree(d_C);
      cudaEventDestroy(iterStart);
      cudaEventDestroy(iterStop);
    }
    cudaDeviceSynchronize();


  }





  
  printf("\n===============================\n");
  printf("Total GPU parallel execution time for %ld iteration(s): %.10f ms\n", niter, total_ms);
  printf("===============================\n");
  

  /*
  if (nrows <= 8 && ncols <= 8) {
    printMatrix(h_A, nrows, ncols, "A");
    printMatrix(h_B, nrows, ncols, "B");
    printMatrix(h_C, nrows, ncols, "C (Result)");
  }
  */  


  free(h_A);
  free(h_B);
  free(h_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  

  
  
  return 0;
}


