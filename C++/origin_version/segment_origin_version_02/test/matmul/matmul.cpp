#include <iostream>
#include <string>
#include <stdio.h>
#include <cuda_runtime.h>
#include "postprocess.h"

int main(){
    // float a[6] = {1,2,3,4,5,6};
    int a_rows = 3;
    int a_cols = 4;
    float a[a_rows * a_cols];
    for (int i = 0; i < a_rows * a_cols; i++){ a[i] = i + 1; }
    for (int i = 0; i < a_rows * a_cols; i++){
        if (i % a_cols == 0) std::cout << std::endl;
        std::cout << a[i] << " ";
    }

    float* a_device = nullptr;
    cudaMalloc(&a_device, a_rows * a_cols * sizeof(float));
    cudaMemcpy(a_device, a, a_rows * a_cols * sizeof(float), cudaMemcpyHostToDevice);

    // float b[12] = {1,2,3,4,4,5,6,7,7,8,9,10};
    int b_rows = 4;
    int b_cols = 6;
    float b[b_rows * b_cols];
    for (int i = 0; i < b_rows * b_cols; i++){ b[i] = i + 4; }
    for (int i = 0; i < b_rows * b_cols; i++){
        if (i % b_cols == 0) std::cout << std::endl;
        std::cout << b[i] << " ";
    }

    float* b_device = nullptr;
    cudaMalloc(&b_device, b_rows * b_cols * sizeof(float));
    cudaMemcpy(b_device, b, b_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice);

    float* c_device = nullptr;
    cudaMalloc(&c_device, a_rows * b_cols * sizeof(float));

    matrix_multiply(a_device, a_rows, a_cols, b_device, b_rows, b_cols, c_device);

    float c[a_rows * b_cols];
    cudaMemcpy(c, c_device, a_rows * b_cols * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < a_rows * b_cols; i++){
        if (i % b_cols == 0) std::cout << std::endl;
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);

    return 0;
}
