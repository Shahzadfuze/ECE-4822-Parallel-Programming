#include <stdio.h>
#include <stdlib.h>



#define N 1000
#define MAX_ERR 1e-6


__global__ void vector_add(float* out, float* a, float* b, int n) {
  for(int i = 0; i < n; i++){
    out[i] = a[i] + b[i];
    printf("%f\n", out[i]);
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
  
  vector_add<<<1, 1>>>(d_out, d_a, d_b, N);
  //  vector_add(out, a, b, N);


  // Error checking to make sure everything works

  // wait for kernel and check for errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
  }


  
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
