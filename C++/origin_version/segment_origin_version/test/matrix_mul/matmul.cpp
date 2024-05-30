#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "postprocess.h"

int main(){
    float proto[18] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    int protoC = 2;  // default 32
    int protoH = 3;  // default 160
    int protoW = 3;  // default 160
    float* protoDevice = nullptr;
    cudaMalloc(&protoDevice, 18 * sizeof(float));
    cudaMemcpy(protoDevice, proto, 18 * sizeof(float), cudaMemcpyHostToDevice);

    // prepare 32 length mask coef space on device
    float maskCoef[2] = {1,2};
    float* maskCoefDevice = nullptr;
    cudaMalloc(&maskCoefDevice, protoC * sizeof(float));
    cudaMemcpy(maskCoefDevice, maskCoef, protoC * sizeof(float), cudaMemcpyHostToDevice);
    // prepare 160x160 mask space on device
    float* maskDevice = nullptr;
    cudaMalloc(&maskDevice, protoH * protoW * sizeof(float));

    process_mask_cuda(protoDevice, maskCoefDevice, maskDevice, protoC, protoH, protoW);

    float mask[protoH * protoW];
    cudaMemcpy(mask, maskDevice, protoH * protoW * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < protoH * protoW; i++){
        std::cout << mask[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}