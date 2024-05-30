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


void process_mask_cuda(float* protoDevice, float* maskCoefDevice, float* maskDevice, int protoC, int protoH, int protoW);


void crop_mask(float* maskDevice, float* downBboxDevice, int maskH, int maskW);


#endif  // POSTPROCESS_H
