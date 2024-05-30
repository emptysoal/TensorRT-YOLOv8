#include "postprocess.h"


__device__ float sigmoid(float data){
    return 1.0f / (1.0f + expf(-data));
}


__global__ void matrix_multiply_kernel(float* aMatrix, int aCols, float* bMatrix, int bCols, float* cMatrix, int cSize, bool sigm){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= cSize) return;

    float target = 0;
    for (int j = 0; j < aCols; j++){
        target += aMatrix[position / bCols * aCols + j] * bMatrix[j * bCols + position % bCols];
    }
    if (sigm) cMatrix[position] = sigmoid(target);
    else cMatrix[position] = target;

}

void matrix_multiply(float* aMatrix, int aRows, int aCols, float* bMatrix, int bRows, int bCols, float* cMatrix, bool sigm){
    int cSize = aRows * bCols;
    int blockSize = 256;
    int gridSize = (cSize + blockSize - 1) / blockSize;
    matrix_multiply_kernel<<<gridSize, blockSize>>>(aMatrix, aCols, bMatrix, bCols, cMatrix, cSize, sigm);
}
