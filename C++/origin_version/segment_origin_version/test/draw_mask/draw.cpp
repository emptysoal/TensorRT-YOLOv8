#include <iostream>
#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "draw.h"


void run(const std::string& imagePath){
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);

    // make mask
    int h = img.rows;
    int w = img.cols;
    float* mask = new float[h * w];
    for (int x = 0; x < w; x++){
        for (int y = 0; y < h; y++){
            int idx = x + y * w;
            if (x > 100 && x < 300 && y > 80 && y < 240) mask[idx] = 0.8f;
            else mask[idx] = 0.2f;
        }
    }

    draw_mask(img, mask);

    float* mask2 = new float[h * w];
    for (int x = 0; x < w; x++){
        for (int y = 0; y < h; y++){
            int idx = x + y * w;
            if (x > 360 && x < 520 && y > 320 && y < 480) mask2[idx] = 0.7f;
            else mask2[idx] = 0.3f;
        }
    }

    draw_mask(img, mask2);

    cv::imwrite("res.jpg", img);

    delete [] mask;
}


int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("This program need 1 argument\n");
        printf("Usage: ./normal [image path]\n");
        printf("Example: ./normal lena.jpg\n");
        return 1;
    }

    std::string imagePath(argv[1]);
    run(imagePath);

    return 0;
}