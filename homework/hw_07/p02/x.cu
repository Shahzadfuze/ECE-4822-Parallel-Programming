#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <fstream>

__global__ void iir_block_kernel(const float* x, float* y,const float* b, const float* a, int M, int N, int numSamples, int blockSize)
{
  extern __shared__ float sh[];
  float* xhist = sh;
  float* yhist = sh + M;

  if (threadIdx.x == 0) {
    for (int i = 0; i < M; ++i) xhist[i] = 0.f;
    for (int j = 0; j < N - 1; ++j) yhist[j] = 0.f;
  }
  __syncthreads();

  int tid = threadIdx.x;
  int start = blockIdx.x * blockSize;
  int end   = min(start + blockSize, numSamples);

  for (int n = start; n < end; ++n) {
    if (tid == 0) {
      // shift histories
      for (int i = M - 1; i > 0; --i) xhist[i] = xhist[i - 1];
      for (int j = N - 2; j > 0; --j) yhist[j] = yhist[j - 1];
      xhist[0] = x[n];
    }
    __syncthreads();

    // parallel feed-forward and feedback
    float ff = 0.f, fb = 0.f;
    for (int i = tid; i < M; i += blockDim.x) ff += b[i] * xhist[i];
    for (int j = tid + 1; j < N; j += blockDim.x) fb += a[j] * yhist[j - 1];

    // reduce within block
    __shared__ float sFF, sFB;
    atomicAdd(&sFF, ff);
    atomicAdd(&sFB, fb);
    __syncthreads();

    if (tid == 0) {
      float yv = (sFF - sFB) / a[0];
      yhist[0] = yv;
      y[n] = yv;
      sFF = 0.f; sFB = 0.f; // reset for next sample
    }
    __syncthreads();
  }
}

int main(int argc, char** argv) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <signal> <bCoef> <aCoef>\n";
    return 1;
  }
  std::ifstream sfile(argv[1]), bfile(argv[2]), afile(argv[3]);
  std::vector<float> x, b, a;
  float v;
  while (sfile >> v) x.push_back(v);
  while (bfile >> v) b.push_back(v);
  while (afile >> v) a.push_back(v);

  int numSamples = x.size(), M = b.size(), N = a.size();
  std::vector<float> y(numSamples, 0.f);

  float *d_x, *d_y, *d_b, *d_a;
  cudaMalloc(&d_x, numSamples*sizeof(float));
  cudaMalloc(&d_y, numSamples*sizeof(float));
  cudaMalloc(&d_b, M*sizeof(float));
  cudaMalloc(&d_a, N*sizeof(float));
  cudaMemcpy(d_x, x.data(), numSamples*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b.data(), M*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_a, a.data(), N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemset(d_y, 0, numSamples*sizeof(float));

  int blockSize = 1024;
  int numBlocks = (numSamples + blockSize - 1) / blockSize;
  size_t shmem = (M + N - 1) * sizeof(float);

  cudaEventRecord(start);  

  iir_block_kernel<<<numBlocks, 128, shmem>>>(d_x, d_y, d_b, d_a, M, N, numSamples, blockSize);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  

  cudaDeviceSynchronize();

  // Measure elapsed time
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %f ms\n", milliseconds);
  

  cudaMemcpy(y.data(), d_y, numSamples*sizeof(float), cudaMemcpyDeviceToHost);
  //  for (float val : y) printf("%f\n", val);

  cudaFree(d_x); cudaFree(d_y); cudaFree(d_b); cudaFree(d_a);

  // Destroy events
  cudaEventDestroy(start);

  cudaEventDestroy(stop);
  
  
  return 1;
}
