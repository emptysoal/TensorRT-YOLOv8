#include <iostream>
#include <string>
#include <stdio.h>
// #include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "postprocess.h"

int main(){

    int mask_h = 8;
    int mask_w = 8;
    int mask_size = mask_h * mask_w;
    float mask[mask_size];
    for (int i = 0; i < mask_size; i++){ mask[i] = i + 1; }
    for (int i = 0; i < mask_size; i++){
        if (i % 8 == 0) std::cout << std::endl;
        std::cout << mask[i] << " ";
    }
    std::cout << std::endl;
    float* maskDevice = nullptr;
    cudaMalloc(&maskDevice, mask_size * sizeof(float));
    cudaMemcpy(maskDevice, mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);

    float bbox[4] = {2, 2, 5, 5};
    float* bboxDevice = nullptr;
    cudaMalloc(&bboxDevice, 4 * sizeof(float));
    cudaMemcpy(bboxDevice, bbox, 4 * sizeof(float), cudaMemcpyHostToDevice);

    crop_mask(maskDevice, bboxDevice, mask_h, mask_w);

    float mask_out[mask_size];
    cudaMemcpy(mask_out, maskDevice, mask_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < mask_size; i++){
        if (i % 8 == 0) std::cout << std::endl;
        std::cout << mask_out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(maskDevice);
    cudaFree(bboxDevice);

    return 0;
}