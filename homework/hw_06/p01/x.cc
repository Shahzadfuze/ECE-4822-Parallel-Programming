#include "x.h"

// This is using a sliding window approch kind of like streaming (going 1 by 1) 


int main(int argc, char** argv){


  if (argc != 3){
    printf("Please use this format <%s> <signal> <coef> >  <output_file>\n", argv[0]);
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

  }


  int n = coef.size();
  if (n == 0) {
    printf("No coefficients loaded!\n");
    return 1;
  }

  // Load all samples into memory
  std::vector<float> samples;
  float s;
  while (signalFile >> s) samples.push_back(s);

  std::vector<float> output(samples.size(), 0.0f);

  const int BLOCK_SIZE = 1024; // Number of samples per block
  int numBlocks = (samples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Parallel block processing
#pragma omp parallel for
  for (int b = 0; b < numBlocks; ++b) {
    int start = b * BLOCK_SIZE;
    int end = std::min(start + BLOCK_SIZE, (int)samples.size());

    // History buffer for this block
    std::vector<float> history(n, 0.0f);
    int circIndex = 0;

    // Initialize history from previous block if not first block
    if (b > 0) {
      int prevEnd = start;
      int copyStart = std::max(0, prevEnd - (n-1));
      int copyCount = prevEnd - copyStart;
      for (int i = 0; i < copyCount; ++i)
	history[i] = samples[copyStart + i];
      circIndex = copyCount % n;
    }

    // Process block
    for (int i = start; i < end; ++i) {
      output[i] = fir_filter_circ(samples[i], coef.data(), n, history.data(), &circIndex);
    }
  }


  int count = 0;
  /*
  // Print output
  for (float y : output){

    //    if (count < 100 ){
      printf("%f\n", y);
      count++;
         }
    else{
      break;
      }


}
  */
  return 0;

}
