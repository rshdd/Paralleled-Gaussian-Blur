#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <chrono>
#include <ctime>
#include "device_launch_parameters.h"
#include <string>

#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        std::cerr << "Error " << cudaGetErrorString(_m_cudaStat)            \
                  << " at line " << __LINE__ << " in file " << __FILE__     \
                  << std::endl;                                             \
        exit(1);                                                            \
    } }

using namespace std;

__global__ void gaussianBlurKernel(const unsigned* input, unsigned* output, int width, int height, int channels, float* filter, int filterWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            for (int i = -filterWidth / 2; i <= filterWidth / 2; i++) {
                for (int j = -filterWidth / 2; j <= filterWidth / 2; j++) {
                    int curRow = min(max(row + i, 0), height - 1);
                    int curCol = min(max(col + j, 0), width - 1);
                    sum += input[(curRow * width + curCol) * channels + c] * filter[(i + filterWidth / 2) * filterWidth + (j + filterWidth / 2)];
                }
            }
            output[(row * width + col) * channels + c] = static_cast<unsigned>(sum);
        }
    }
}

void gaussianBlur(const cv::Mat& input, cv::Mat& output, float* filter, int filterWidth) {
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t imageSize = width * height * channels * sizeof(unsigned);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);

    unsigned* d_input;
    unsigned* d_output;
    float* d_filter;

    CUDA_CHECK_RETURN(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_filter, filterSize));

    CUDA_CHECK_RETURN(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    gaussianBlurKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, channels, d_filter, filterWidth);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    CUDA_CHECK_RETURN(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK_RETURN(cudaFree(d_input));
    CUDA_CHECK_RETURN(cudaFree(d_output));
    CUDA_CHECK_RETURN(cudaFree(d_filter));
}

int main() {
    // Read input image
    cv::Mat input = cv::imread("input.jpg", cv::IMREAD_COLOR);
    if (input.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }

    // Define Gaussian filter
    int filterWidth = 5;
    float filter[] = {
        1,  4,  7,  4, 1,
        4, 16, 26, 16, 4,
        7, 26, 41, 26, 7,
        4, 16, 26, 16, 4,
        1,  4,  7,  4, 1
    };
    for (int i = 0; i < filterWidth * filterWidth; i++) {
        filter[i] /= 273.0f;
    }

    cv::Mat output(input.size(), input.type());

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();
    gaussianBlur(input, output, filter, filterWidth);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds" << std::endl;

    // Save output image
    cv::imwrite("output.jpg", output);

    return 0;
}