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


// ------------------ cut mask --------------------
__global__ void cut_mask_kernel(
    float* masks, int maskNum, int maskHeight, int maskWidth,
    float* cutMasks, int cutMaskTop, int cutMaskLeft, int cutMaskH, int cutMaskW
){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  // x in global threads
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int cutMaskWidthTotal = maskNum * cutMaskW;
    if (ix >= cutMaskWidthTotal || iy >= cutMaskH) return;

    int maskIdx = ix / cutMaskW;

    float* presentCutMask = cutMasks + maskIdx * cutMaskH * cutMaskW;
    float* presentMask = masks + maskIdx * maskHeight * maskWidth;

    int ix_present = ix % cutMaskW;
    int idx = ix_present + iy * cutMaskW;

    int src_ix = ix_present + cutMaskLeft;
    int src_iy = iy + cutMaskTop;
    int src_idx = src_ix + src_iy * maskWidth;

    presentCutMask[idx] = presentMask[src_idx];
}

void cut_mask(
    float* masksDevice, int maskNum, int maskHeight, int maskWidth,
    float* cutMasksDevice, int cutMaskTop, int cutMaskLeft, int cutMaskH, int cutMaskW
){
    int cutMaskWidthTotal = maskNum * cutMaskW;
    dim3 blockSize(32, 32);
    dim3 gridSize((cutMaskWidthTotal + blockSize.x - 1) / blockSize.x, (cutMaskH + blockSize.y - 1) / blockSize.y);
    cut_mask_kernel<<<gridSize, blockSize>>>(masksDevice, maskNum, maskHeight, maskWidth,
                                            cutMasksDevice, cutMaskTop, cutMaskLeft, cutMaskH, cutMaskW);
}


// ------------------ bilinear resize mask --------------------
__global__ void resize_kernel(float* masks, int maskNum, int maskHeight, int maskWidth, float* dstMasks, int dstMaskH, int dstMaskW){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;  // x in global threads
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    int dstMaskWidthTotal = maskNum * dstMaskW;
    if (ix >= dstMaskWidthTotal || iy >= dstMaskH) return;

    int maskIdx = ix / dstMaskW;

    float* srcMask = masks + maskIdx * maskHeight * maskWidth;
    float* dstMask = dstMasks + maskIdx * dstMaskH * dstMaskW;

    ix = ix % dstMaskW;
    int idx = ix + iy * dstMaskW;

    float scaleY = (float)dstMaskH / (float)maskHeight;
    float scaleX = (float)dstMaskW / (float)maskWidth;

    // (ix, iy)为目标图像坐标
    // (before_x, before_y)为原图坐标
    float beforeX = float(ix + 0.5) / scaleX - 0.5;
    float beforeY = float(iy + 0.5) / scaleY - 0.5;
    // 原图像坐标四个相邻点
    // 获得变换前最近的四个顶点,取整
    int topY = static_cast<int>(beforeY);
    int bottomY = topY + 1;
    int leftX = static_cast<int>(beforeX);
    int rightX = leftX + 1;
    //计算变换前坐标的小数部分
    float u = beforeX - leftX;
    float v = beforeY - topY;

    if (topY >= maskHeight - 1){  // 对应原图的坐标位于最后一行
        topY = maskHeight - 1;
        bottomY = maskHeight - 1;
    }
    if (leftX >= maskWidth - 1){  // 对应原图的坐标位于最后一列
        leftX = maskWidth - 1;
        rightX = maskWidth - 1;
    }

    dstMask[idx] = (1. - u) * (1. - v) * srcMask[leftX + topY * maskWidth]
                 + (u) * (1. - v) * srcMask[rightX + topY * maskWidth]
                 + (1. - u) * (v) * srcMask[leftX + bottomY * maskWidth]
                 + u * v * srcMask[rightX + bottomY * maskWidth];
}

void resize(float* masksDevice, int maskNum, int maskHeight, int maskWidth, float* dstMasksDevice, int dstMaskH, int dstMaskW){
    int dstMaskWidthTotal = maskNum * dstMaskW;
    dim3 blockSize(32, 32);
    dim3 gridSize((dstMaskWidthTotal + blockSize.x - 1) / blockSize.x, (dstMaskH + blockSize.y - 1) / blockSize.y);
    resize_kernel<<<gridSize, blockSize>>>(masksDevice, maskNum, maskHeight, maskWidth, dstMasksDevice, dstMaskH, dstMaskW);
}
