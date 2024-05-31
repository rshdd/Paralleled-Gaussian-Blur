#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using namespace std;

#define CUDA_CHECK_RETURN(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        std::cerr << "Error " << cudaGetErrorString(_m_cudaStat)            \
                  << " at line " << __LINE__ << " in file " << __FILE__     \
                  << std::endl;                                             \
        exit(1);                                                            \
    } }

__global__ void gaussianBlurKernel(const unsigned char* input, unsigned char* output, int width, int height, int channels, float* filter, int filterWidth) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        for (int c = 0; c < channels; c++) {
            float sum = 0.0f;
            float normalization = 0.0f;
            for (int i = -filterWidth / 2; i <= filterWidth / 2; i++) {
                for (int j = -filterWidth / 2; j <= filterWidth / 2; j++) {
                    int curRow = min(max(row + i, 0), height - 1);
                    int curCol = min(max(col + j, 0), width - 1);
                    float filterValue = filter[(i + filterWidth / 2) * filterWidth + (j + filterWidth / 2)];
                    sum += input[(curRow * width + curCol) * channels + c] * filterValue;
                    normalization += filterValue;
                }
            }
            output[(row * width + col) * channels + c] = static_cast<unsigned char>(sum / normalization);
        }
    }
}

void gaussianBlur(const cv::Mat& input, cv::Mat& output, float* filter, int filterWidth) {
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();
    size_t imageSize = width * height * channels * sizeof(unsigned char);
    size_t filterSize = filterWidth * filterWidth * sizeof(float);

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    float* d_filter = nullptr;

    // Allocate memory on the GPU
    CUDA_CHECK_RETURN(cudaMalloc(&d_input, imageSize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_output, imageSize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_filter, filterSize));

    // Copy input data from host to device
    CUDA_CHECK_RETURN(cudaMemcpy(d_input, input.data, imageSize, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice));

    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Measure time for CUDA execution
    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernel
    gaussianBlurKernel << <gridSize, blockSize >> > (d_input, d_output, width, height, channels, d_filter, filterWidth);
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cuda = end - start;
    std::cout << "Time taken to apply Gaussian blur using CUDA: " << elapsed_cuda.count() << " seconds" << std::endl;

    // Copy result from device to host
    CUDA_CHECK_RETURN(cudaMemcpy(output.data, d_output, imageSize, cudaMemcpyDeviceToHost));

    // Free allocated memory on the GPU
    if (d_input) cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_filter) cudaFree(d_filter);
}

void gaussianBlurSequential(const cv::Mat& input, cv::Mat& output, const float* filter, int filterWidth) {
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();

    auto start = std::chrono::high_resolution_clock::now();

    // Iterate over each pixel in the image
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Initialize sum and normalization variables for each channel
            float sum[3] = { 0.0f, 0.0f, 0.0f };
            float normalization = 0.0f;

            // Iterate over the filter window
            for (int fy = -filterWidth / 2; fy <= filterWidth / 2; fy++) {
                for (int fx = -filterWidth / 2; fx <= filterWidth / 2; fx++) {
                    // Calculate coordinates in the input image
                    int ix = std::min(std::max(x + fx, 0), width - 1);
                    int iy = std::min(std::max(y + fy, 0), height - 1);

                    // Compute the filter value
                    float filterValue = filter[(fy + filterWidth / 2) * filterWidth + (fx + filterWidth / 2)];

                    // Accumulate the weighted sum for each channel
                    for (int c = 0; c < channels; c++) {
                        sum[c] += input.at<cv::Vec3b>(iy, ix)[c] * filterValue;
                    }

                    // Accumulate normalization factor
                    normalization += filterValue;
                }
            }

            // Normalize and set the output pixel value
            for (int c = 0; c < channels; c++) {
                output.at<cv::Vec3b>(y, x)[c] = static_cast<unsigned char>(sum[c] / normalization);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_sequential = end - start;
    std::cout << "Time taken to apply Gaussian blur sequentially: " << elapsed_sequential.count() << " seconds" << std::endl;
}

int main() {
    // Open the default camera
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening camera" << std::endl;
        return -1;
    }

    // Define Gaussian filter
    int filterWidth = 7; // Example filter size
    float filter[] = {
        2, 4, 2,
        4, 8, 4,
        2, 4, 2
    };

    while (true) {
        // Measure time for each frame
        auto start_frame = std::chrono::high_resolution_clock::now();

        // Capture frame-by-frame
        cv::Mat frame;
        cap >> frame;

        // Check if the frame is empty
        if (frame.empty()) {
            std::cerr << "Error: Could not read frame from the camera." << std::endl;
            break;
        }

        // Create output images
        cv::Mat output_cuda(frame.size(), frame.type());
        cv::Mat output_sequential(frame.size(), frame.type());

        // Apply Gaussian blur using CUDA
        gaussianBlur(frame, output_cuda, filter, filterWidth);

        // Apply Gaussian blur sequentially
        //gaussianBlurSequential(frame, output_sequential, filter, filterWidth);

        // Display original and blurred images
        cv::imshow("Original", frame);
        cv::imshow("Blurred (CUDA)", output_cuda);
        //cv::imshow("Blurred (Sequential)", output_sequential);

        // Check for 'q' key press to exit loop
        if (cv::waitKey(1) == 'q') {
            break;
        }

        // Measure time for the current frame
        auto end_frame = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_frame = end_frame - start_frame;
        std::cout << "Time taken for this frame: " << elapsed_frame.count() << " seconds" << std::endl;
    }

    // Release the camera
    cap.release();
    cv::destroyAllWindows();

    return 0;
}
