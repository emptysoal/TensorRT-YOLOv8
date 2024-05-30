#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

void transpose(float* src, float* dst, int numBboxes, int numElements);
/*
    transpose [56 8400] convert to [8400 56]
src:          Tensor, dim is [56 8400]
dst:          Tensor, dim is [8400 56]
numBboxes:    number of bboxes: default 8400
numElements:  center_x, center_y, width, height, 1 classes, 51 key points
*/

void decode(float* src, float* dst, int numBboxes, int numClasses, int numKpts, float confThresh, int maxObjects, int numBoxElement);
/*
    convert [8400 56] to [58001, ], 58001 = 1 + 1000 * (4bbox + cond + cls + keepflag + 51kpts), 1: number of valid bboxes
     1000: max bboxes, valid bboxes may less than 1000, 4bbox: left, top, right, bottom)
*/

void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement);


#endif  // POSTPROCESS_H
