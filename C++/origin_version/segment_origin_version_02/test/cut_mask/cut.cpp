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

    int top = 4;
    int left = 3;
    int cut_rows = 4;
    int cut_cols = 6;
    int cut_size = num * cut_rows * cut_cols;
    float cut_masks[cut_size];

    float* cutMasksDevice = nullptr;
    cudaMalloc(&cutMasksDevice, cut_size * sizeof(float));

    cut_mask(masksDevice, num, rows, cols, cutMasksDevice, top, left, cut_rows, cut_cols);

    cudaMemcpy(cut_masks, cutMasksDevice, cut_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < cut_size; i++){
        if (i % cut_cols == 0) std::cout << std::endl;
        if (i % (cut_rows * cut_cols) == 0) std::cout << std::endl;
        std::cout << cut_masks[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(masksDevice);
    cudaFree(cutMasksDevice);

    return 0;
}
