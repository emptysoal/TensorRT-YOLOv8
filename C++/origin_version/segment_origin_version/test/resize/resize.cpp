#include <iostream>
#include <string>
#include <stdio.h>
// #include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "postprocess.h"

int main(){

    int mask_h = 4;
    int mask_w = 4;
    int mask_size = mask_h * mask_w;
    float mask[mask_size];
    for (int i = 0; i < mask_size; i++){ mask[i] = i + 11; }
    for (int i = 0; i < mask_size; i++){
        if (i % mask_w == 0) std::cout << std::endl;
        std::cout << mask[i] << " ";
    }
    std::cout << std::endl;
    float* maskDevice = nullptr;
    cudaMalloc(&maskDevice, mask_size * sizeof(float));
    cudaMemcpy(maskDevice, mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);

    int dst_h = 7;
    int dst_w = 7;
    int dst_size = dst_h * dst_w;
    float* scaledMaskDevice = nullptr;
    cudaMalloc(&scaledMaskDevice, dst_size * sizeof(float));

    resize(maskDevice, mask_h, mask_w, scaledMaskDevice, dst_h, dst_w);

    float out[dst_size];
    cudaMemcpy(out, scaledMaskDevice, dst_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < dst_size; i++){
        if (i % dst_w == 0) std::cout << std::endl;
        std::cout << (int)out[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(maskDevice);
    cudaFree(scaledMaskDevice);

    return 0;
}