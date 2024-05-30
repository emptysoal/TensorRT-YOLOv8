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
    for (int i = 0; i < mask_size; i++){ mask[i] = i + 11; }
    for (int i = 0; i < mask_size; i++){
        if (i % 8 == 0) std::cout << std::endl;
        std::cout << mask[i] << " ";
    }
    std::cout << std::endl;
    float* maskDevice = nullptr;
    cudaMalloc(&maskDevice, mask_size * sizeof(float));
    cudaMemcpy(maskDevice, mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);

    int cutMaskLeft = 0;
    int cutMaskTop = 2;
    int cutMaskRight = 8;
    int cutMaskBottom = 6;
    int cutMaskWidth = cutMaskRight - cutMaskLeft;
    int cutMaskHeight = cutMaskBottom - cutMaskTop;
    float* cutMaskDevice = nullptr;
    cudaMalloc(&cutMaskDevice, cutMaskHeight * cutMaskWidth * sizeof(float));

    cut_mask(maskDevice, cutMaskDevice, cutMaskTop, cutMaskLeft, cutMaskHeight, cutMaskWidth, mask_w);

    float out[cutMaskHeight * cutMaskWidth];
    cudaMemcpy(out, cutMaskDevice, cutMaskHeight * cutMaskWidth * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < cutMaskHeight * cutMaskWidth; i++){
        if (i % cutMaskWidth == 0) std::cout << std::endl;
        std::cout << out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(maskDevice);
    cudaFree(cutMaskDevice);

    return 0;
}