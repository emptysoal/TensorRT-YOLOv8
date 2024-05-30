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


void crop_mask(float* masksDevice, int maskNum, int maskHeight, int maskWidth, float* bboxesDevice);
/*
    set value 0 where masks out of bboxes
masksDevice:      mask array on device, shape(n, 160 x 160)
maskNum:          n, number of masks
maskHeight:       height of each mask
maskWidth:        width of each mask
bboxesDevice:     bbox array on device, shape(n, 4), 4 : x1, y1, x2, y2
*/

#endif  // POSTPROCESS_H
