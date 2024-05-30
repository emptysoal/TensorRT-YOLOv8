#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

void transpose(float* src, float* dst, int numBboxes, int numElements);
/*
    transpose [1 84 8400] convert to [1 8400 84]
src:          Tensor, dim is [1 84 8400]
dst:          Tensor, dim is [1 8400 84]
numBboxes:    number of bboxes
numElements:  center_x, center_y, width, height, 80 or other classes
*/

void decode(float* src, float* dst, int numBboxes, int numClasses, float confThresh, int maxObjects, int numBoxElement);
/*
    convert [1 8400 84] to [1 7001](7001 = 1 + 1000 * 7, 1: number of valid bboxes
     1000: max bboxes, valid bboxes may less than 1000, 7: left, top, right, bottom, confidence, class, keepflag)
*/

void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement);


#endif  // POSTPROCESS_H
