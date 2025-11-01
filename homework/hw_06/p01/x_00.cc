#include "x.h"



float fir_filter(float input, float *coef, int n, float *history){
  int i;
  float* hist_ptr, *hist1_ptr, *coef_ptr;
  float output;
  
  hist_ptr = history;
  hist1_ptr = hist_ptr;
  coef_ptr = coef + n - 1;

  output = *hist_ptr++ * (*coef_ptr--);

  for(i = 2; i < n; i ++){
    *hist1_ptr++ =  *hist_ptr; // Update history array;
    output += (*hist_ptr++) * (*coef_ptr--);
    
  }
  
  output += input * (*coef_ptr);
  *hist1_ptr = input;

  return(output);

}



// FIR filter using a circular buffer
float fir_filter_circ(float input, const float* coef, int n, float* history, int* index) {
  history[*index] = input; // Insert new sample at current position

  float output = 0.0f;

  // Compute convolution
  int histIdx = *index;
  for(int i = 0; i < n; ++i) {
    output += coef[i] * history[histIdx];
    // Move backwards in the circular buffer
    histIdx = (histIdx - 1 + n) % n;
  }

  // Move index forward for next sample
  *index = (*index + 1) % n;

  return output;
}
