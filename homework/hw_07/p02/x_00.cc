#include "x.h"



// Direct Form I IIR filter
float iir_filter(float input, const std::vector<float>& b, const std::vector<float>& a,
		 std::vector<float>& x_hist, std::vector<float>& y_hist) {

  int M = b.size();
  int N = a.size();

  // Shift input history
  for(int i = M-1; i > 0; --i)
    x_hist[i] = x_hist[i-1];
  x_hist[0] = input;

  // Compute feedforward
  float y = 0.0f;
  for(int i = 0; i < M; ++i)
    y += b[i] * x_hist[i];

  // Compute feedback (skip a[0])
  for(int j = 1; j < N; ++j)
    y -= a[j] * y_hist[j-1];

  // Shift output history
  for(int i = N-1; i > 0; --i)
    y_hist[i] = y_hist[i-1];
  y_hist[0] = y;

  return y;
}
