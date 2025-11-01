#ifndef X_H
#define X_H


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstdlib> 
#include <omp.h>
#include <vector>



// Direct Form I IIR filter
float iir_filter(float input, const std::vector<float>& b, const std::vector<float>& a,
		 std::vector<float>& x_hist, std::vector<float>& y_hist);





#endif // X_H
