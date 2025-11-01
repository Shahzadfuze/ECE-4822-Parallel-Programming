#include <stdio.h>
#include <stdlib.h>



#define N 10000000



// So this doesn't run in parallel becuase we are just going through and doing the
// calculation 1 by 1 hence the forloop 
/*
__global__ void vector_add(float* out, float* a, float* b, int n) {
  int index = threadIdx.x;
  int stride = blockDim.x;

  for(int i = index; i < n; i += stride){
    out[i] = a[i] + b[i];
    printf("%f\n", out[i]);
  }
}

*/

// So to run it in parallel we have to unitlize the thread and blocks
// you calculate the index of a block like a 1d matrix idx = threadsIdx.x * blockDim.x + threadIdx.x


__global__ void vector_add(float* out, float* a, float* b, int n){
  int tid = blockIdx.x * blockDim.x + threadIdx.x; 
  if(tid < n){
    out[tid] = a[tid] + b[tid];
  }
}


int main() {
  float *a, *b, *out;
  float *d_a, *d_b, *d_out;


  // Allocate Memory in the CPU domain
  a =   (float*)malloc(sizeof(float) * N );
  b =   (float*)malloc(sizeof(float) * N );
  out =   (float*)malloc(sizeof(float) * N );


  // Allocate memory in the GPU domain
  cudaMalloc((void**)&d_a, sizeof(float) *N); 
  cudaMalloc((void**)&d_b, sizeof(float) *N); 
  cudaMalloc((void**)&d_out, sizeof(float) *N); 
  
  // Populate array
  for(int i = 0; i < N; i++){
    a[i] = 1.0f; b[i] = 2.0f;
  }


  // Transfer the data from CPU to GPU

  cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);


  int block_size = 256;
  int grid_size = ((N + block_size) / block_size);
  
  vector_add<<<grid_size, block_size>>>(d_out, d_a, d_b, N);


  cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
  
    // Check first 10 results
  for (int i = 0; i < 10; i++) {
    printf("out[%d] = %f\n", i, out[i]);
  }


  //Free GPU
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);
  
 
  // Free CPU
  free(a);
  free(b);
  free(out);

  
  
  return 0;
}
