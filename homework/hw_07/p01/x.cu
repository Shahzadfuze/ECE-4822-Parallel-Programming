
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdlib> 
#include <omp.h>
#include <vector>
#include <cuda_runtime.h>

__global__ void fir_filter_cuda(const float* d_samples, const float* d_coef, float* d_output, int numSamples, int numCoef) {


  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numSamples) return;

  float acc = 0.0f;

  // Perform convolution for output[idx]
  for (int j = 0; j < numCoef; ++j) {
    int sampleIdx = idx - j;
    if (sampleIdx >= 0)
      acc += d_coef[j] * d_samples[sampleIdx];
  }
  d_output[idx] = acc;
}


int main(int argc, char** argv){
  // Create CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (argc != 3){

    printf("Please use this format <%s> <signal> <coef> > <output_file>\n", argv[0]);
    return 1;
  }

  
  std::ifstream signalFile(argv[1]);
  std::ifstream coefFile(argv[2]);

  if (!signalFile.is_open() || !coefFile.is_open()) {
    printf("Error opening file(s).\n");
    return 1;
  }
  

  // Load coefficients
  std::vector<float> coef;
  float c;
  while (coefFile >> c) {
    coef.push_back(c);
    //    printf("Coefs: %f\n", c);
  }

  int n = coef.size();
  if (n == 0) {
    printf("No coefficients loaded!\n");
    return 1;
  }

  
  
  // Load all samples into memory
  std::vector<float> samples;
  float s;
  while (signalFile >> s){



   samples.push_back(s);
   //   printf("Input: %f\n", s);
  }
  std::vector<float> output(samples.size(), 0.0f);
  std::vector<float> history(coef.size(), 0.0f);


  int numSamples = samples.size();
  int numCoef = coef.size();
  

  // Allocate device memory
  float *d_samples, *d_coef, *d_output;
  cudaMalloc((void**)&d_samples, numSamples * sizeof(float));
  cudaMalloc((void**)&d_coef, numCoef * sizeof(float));
  cudaMalloc((void**)&d_output, numSamples* sizeof(float));
  
  // Copy data to device
  cudaMemcpy(d_samples, samples.data(), numSamples * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_coef, coef.data(), numCoef * sizeof(float), cudaMemcpyHostToDevice);

  
  cudaEventRecord(start);

  // Launch enough threads
  int threadsPerBlock = 256;
  int blocks = (numSamples + threadsPerBlock - 1) / threadsPerBlock;
  fir_filter_cuda<<<blocks, threadsPerBlock>>>(d_samples, d_coef, d_output, numSamples, numCoef);
  // Stop timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaDeviceSynchronize();  

  cudaMemcpy(output.data(), d_output, numSamples * sizeof(float), cudaMemcpyDeviceToHost);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("CUDA kernel execution time: %f ms\n", milliseconds);

  // Print results
  /*
  for (int i = 0; i < numSamples; ++i)
        printf("%f\n", output[i]);
  */
 
  // Cleanup
  cudaFree(d_samples);
  cudaFree(d_coef);
  cudaFree(d_output);
  

  return 0;

   
}
