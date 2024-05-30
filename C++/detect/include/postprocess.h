#ifndef POSTPROCESS_H
#define POSTPROCESS_H

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "config.h"

void transpose(float* src, float* dst, int numBboxes, int numElements, cudaStream_t stream);
/*
    transpose [1 84 8400] convert to [1 8400 84]
src:          Tensor, dim is [1 84 8400]
dst:          Tensor, dim is [1 8400 84]
numBboxes:    number of bboxes
numElements:  center_x, center_y, width, height, 80 or other classes
*/

void decode(float* src, float* dst, int numBboxes, int numClasses, float confThresh, int maxObjects, int numBoxElement, cudaStream_t stream);
/*
    convert [1 8400 84] to [1 7001](7001 = 1 + 1000 * 7, 1: number of valid bboxes
     1000: max bboxes, valid bboxes may less than 1000, 7: left, top, right, bottom, confidence, class, keepflag)
*/

void nms(float* data, float kNmsThresh, int maxObjects, int numBoxElement, cudaStream_t stream);


__inline__ void scale_bbox(cv::Mat& img, float bbox[4]){
    float r_w = kInputW / (img.cols * 1.0);
    float r_h = kInputH / (img.rows * 1.0);
    float r = std::min(r_w, r_h);
    float pad_h = (kInputH - r * img.rows) / 2;
    float pad_w = (kInputW - r * img.cols) / 2;

    bbox[0] = (bbox[0] - pad_w) / r;
    bbox[1] = (bbox[1] - pad_h) / r;
    bbox[2] = (bbox[2] - pad_w) / r;
    bbox[3] = (bbox[3] - pad_h) / r;
}


#endif  // POSTPROCESS_H
