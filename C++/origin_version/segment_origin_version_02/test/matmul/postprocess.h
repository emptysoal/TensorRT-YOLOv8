#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cmath>
#include <cuda_runtime.h>

void matrix_multiply(float* aMatrix, int aRows, int aCols, float* bMatrix, int bRows, int bCols, float* cMatrix, bool sigm = false);
/*
    matrix multiply, like numpy.matmul() function
aMatrix:          input matrix 1 array on device
aRows:            rows of input matrix 1
aCols:            columns of input matrix 1
bMatrix:          input matrix 2 array on device
bRows:            rows of input matrix 2
bCols:            columns of input matrix 2
cMatrix:          output matrix array on device
sigm:             Whether to do sigmoid() on the result 
*/

#endif  // POSTPROCESS_H
