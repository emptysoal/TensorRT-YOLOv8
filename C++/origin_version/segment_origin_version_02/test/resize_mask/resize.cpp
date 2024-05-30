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

    int dst_rows = 15;
    int dst_cols = 15;
    int dst_size = num * dst_rows * dst_cols;
    float dst_masks[dst_size];

    float* dstMasksDevice = nullptr;
    cudaMalloc(&dstMasksDevice, dst_size * sizeof(float));

    resize(masksDevice, num, rows, cols, dstMasksDevice, dst_rows, dst_cols);

    cudaMemcpy(dst_masks, dstMasksDevice, dst_size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < dst_size; i++){
        if (i % dst_cols == 0) std::cout << std::endl;
        if (i % (dst_rows * dst_cols) == 0) std::cout << std::endl;
        std::cout << (int)dst_masks[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(masksDevice);
    cudaFree(dstMasksDevice);

    return 0;
}
