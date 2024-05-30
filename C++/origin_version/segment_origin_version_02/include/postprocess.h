#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <cmath>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

void transpose(float* src, float* dst, int numBboxes, int numElements);
/*
    transpose [116 8400] convert to [8400 116]
src:          Tensor, dim is [116 8400]
dst:          Tensor, dim is [8400 116]
numBboxes:    number of bboxes: default 8400
numElements:  center_x, center_y, width, height, 80 classes, 32 masks
*/

void decode(float* src, float* dst, int numBboxes, int numClasses, int numMasks, float confThresh, int maxObjects, int numBoxElement);
/*
    convert [8400 116] to [39001, ], 39001 = 1 + 1000 * (4bbox + cond + cls + keepflag + 32masks), 1: number of valid bboxes
     1000: max bboxes, valid bboxes may less than 1000, 4bbox: left, top, right, bottom)
*/

void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement);


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


void downsample_bbox(float* bboxDevice, int length, float heightRatio, float widthRatio);


void crop_mask(float* masksDevice, int maskNum, int maskHeight, int maskWidth, float* bboxesDevice);
/*
    set value 0 where masks out of bboxes
masksDevice:      mask array on device, shape(n, 160 x 160)
maskNum:          n, number of masks
maskHeight:       height of each mask
maskWidth:        width of each mask
bboxesDevice:     bbox array on device, shape(n, 4), 4 : x1, y1, x2, y2
*/


void cut_mask(
    float* masksDevice, int maskNum, int maskHeight, int maskWidth,
    float* cutMasksDevice, int cutMaskTop, int cutMaskLeft, int cutMaskH, int cutMaskW
);


void resize(float* masksDevice, int maskNum, int maskHeight, int maskWidth, float* dstMasksDevice, int dstMaskH, int dstMaskW);


#endif  // POSTPROCESS_H
