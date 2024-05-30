#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
#include "postprocess.h"

int main(){
    int num = 4;
    int rows = 10;
    int cols = 10;
    int size = num * rows * cols;
    float masks[size];
    for (int i = 0; i < size; i++){ masks[i] = i + 100; }
    for (int i = 0; i < size; i++){
        if (i % cols == 0) std::cout << std::endl;
        if (i % (rows * cols) == 0) std::cout << std::endl;
        std::cout << masks[i] << " ";
    }

    float* masksDevice = nullptr;
    cudaMalloc(&masksDevice, size * sizeof(float));
    cudaMemcpy(masksDevice, masks, size * sizeof(float), cudaMemcpyHostToDevice);

    int bbox_cols = 4;
    float bboxes[num * bbox_cols] = {2,2,4,4,3,4,6,6,5,3,8,7,2,5,4,8};
    // float bboxes[num * bbox_cols] = {2,2,5,4,6,5,8,8};
    // for (int i = 0; i < b_rows * b_cols; i++){ b[i] = i + 4; }
    for (int i = 0; i < num * bbox_cols; i++){
        if (i % bbox_cols == 0) std::cout << std::endl;
        std::cout << bboxes[i] << " ";
    }

    float* bboxesDevice = nullptr;
    cudaMalloc(&bboxesDevice, num * bbox_cols * sizeof(float));
    cudaMemcpy(bboxesDevice, bboxes, num * bbox_cols * sizeof(float), cudaMemcpyHostToDevice);

    crop_mask(masksDevice, num, rows, cols, bboxesDevice);

    cudaMemcpy(masks, masksDevice, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++){
        if (i % cols == 0) std::cout << std::endl;
        if (i % (rows * cols) == 0) std::cout << std::endl;
        std::cout << masks[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(masksDevice);
    cudaFree(bboxesDevice);

    return 0;
}
