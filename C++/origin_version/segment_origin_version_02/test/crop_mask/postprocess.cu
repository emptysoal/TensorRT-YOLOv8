#include "postprocess.h"


// ------------------ matrix multiply --------------------
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


// ------------------ crop mask --------------------
__global__ void crop_mask_kernel(float* masks, int maskNum, int maskHeight, int maskWidth, float* bboxes){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  // x in global threads
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int maskWidthTotal = maskNum * maskWidth;
    if (ix >= maskWidthTotal || iy >= maskHeight) return;

    int maskIdx = ix / maskWidth;

    float* presentMask = masks + maskIdx * maskHeight * maskWidth;
    float* presentBbox = bboxes + maskIdx * 4;

    int ix_present = ix % maskWidth;
    int idx = ix_present + iy * maskWidth;
    if (ix_present < presentBbox[0] || ix_present > presentBbox[2] || iy < presentBbox[1] || iy > presentBbox[3]){
        presentMask[idx] = 0.0f;
    }
}

void crop_mask(float* masksDevice, int maskNum, int maskHeight, int maskWidth, float* bboxesDevice){
    // alloc 2D thread shape (h, w), h >= maskHeight, w >= maskNum * maskWidth
    // that is to say: view 2D mask as horizontal mode
    int maskWidthTotal = maskNum * maskWidth;
    dim3 blockSize(32, 32);
    dim3 gridSize((maskWidthTotal + blockSize.x - 1) / blockSize.x, (maskHeight + blockSize.y - 1) / blockSize.y);
    crop_mask_kernel<<<gridSize, blockSize>>>(masksDevice, maskNum, maskHeight, maskWidth, bboxesDevice);
}
