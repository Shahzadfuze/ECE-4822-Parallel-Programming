#include "x.h"

// This is using a sliding window approch kind of like streaming (going 1 by 1) 


int main(int argc, char** argv){


  if (argc != 4){
    printf("Please use this format <%s> <signal> <bCoef> <aCoef> >  <output_file>\n", argv[0]);
    return 1;
  }

  std::ifstream signalFile(argv[1]);
  std::ifstream bFile(argv[2]);
  std::ifstream aFile(argv[3]);

  
  if (!signalFile.is_open() || !bFile.is_open() || !aFile.is_open()) {
    printf("Error opening file(s).\n");
    return 1;
  }
  
  // Load coefficients
  std::vector<float> bCoef;
  float b;
  while (bFile >> b) {

    bCoef.push_back(b);
    //    printf("B: %f\n", b);
  }

  std::vector<float> aCoef;
  float a;
  while(aFile >> a) {
    
    aCoef.push_back(a);
    //    printf("A: %f\n", a);
  }

  
  int n = bCoef.size();
  if (n == 0) {
    printf("No coefficients loaded!\n");
    return 1;
  }

  int m = aCoef.size();
  if(n == 0){
    printf("No coefficients loaded!\n");
    return 1;
  }

  // Input history (length = bCoef.size())
  std::vector<float> xHist(bCoef.size(), 0.0f);

  // Output history (length = aCoef.size() - 1, because a[0] is used in formula)
  std::vector<float> yHist(aCoef.size() - 1, 0.0f);


  // Load all the samples into memory
  std::vector<float> samples;
  float s;
  while(signalFile >> s) samples.push_back(s);

  std::vector<float> output(samples.size(), 0.0f);

  const int BLOCK_SIZE = 1024;
  int numBlocks = (samples.size() + BLOCK_SIZE - 1) / BLOCK_SIZE;


  // Precompute histories for each block
  std::vector<std::vector<float>> xHistories(numBlocks, std::vector<float>(bCoef.size(), 0.0f));
  std::vector<std::vector<float>> yHistories(numBlocks, std::vector<float>(aCoef.size()-1, 0.0f));

  

  // Copy last samples from previous block for history initialization
  for(int b = 1; b < numBlocks; ++b){
    int prevEnd = b*BLOCK_SIZE;
    int copyCountX = std::min((int)bCoef.size()-1, prevEnd);
    int copyCountY = std::min((int)aCoef.size()-1, prevEnd);

    for(int i = 0; i < copyCountX; ++i)
      xHistories[b][i] = samples[prevEnd - copyCountX + i];

    for(int i = 0; i < copyCountY; ++i)
      yHistories[b][i] = output[prevEnd - copyCountY + i]; // initially zero for first iteration
  }


  // Process each block in parallel
      #pragma omp parallel for
  for(int b = 0; b < numBlocks; ++b){
    int start = b * BLOCK_SIZE;
    int end = std::min(start + BLOCK_SIZE, (int)samples.size());

    auto& xHist = xHistories[b];
    auto& yHist = yHistories[b];

    for(int i = start; i < end; ++i){
      output[i] = iir_filter(samples[i], bCoef, aCoef, xHist, yHist);
    }
  }


  /*
  // Print filtered output
  for(float y : output)
    printf("%f\n", y);
  */


  
  
  /*
  float sample;
  while(signalFile >> sample) {
    float y = iir_filter(sample, bCoef, aCoef, xHist, yHist);
    printf("%f\n", y); // Or store in an output vector
  }
  */

  
  return 0;

}
