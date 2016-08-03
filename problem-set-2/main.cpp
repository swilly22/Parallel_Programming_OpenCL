//
//  main.cpp
//  opencv-blur
//
//

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#include "blurkernel.cl.h"

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char * argv[]) {
    if(argc != 2) {
        printf("Missing path to image file.");
        return -1;
    }
    
    // Load image from disk.
    const char* imagePath = argv[1];
    Mat image;
    image = imread(imagePath, CV_LOAD_IMAGE_COLOR);
    
    if(! image.data) {
        printf("Could not open or find the image");
        return -1;
    }
    
    // Determin number of channels in image.
    int channels = image.channels();
    
    // Compute image width, take into account number of channels.
    size_t imageWidth = image.cols * channels;
    size_t imageHeight = image.rows;
    
    // Does pixels data stored continuously?
    if(image.isContinuous()) {
        // Update image width.
        imageWidth *= imageHeight;
        imageHeight = 1;
    }
    
    // Create window
    namedWindow("Display window", WINDOW_AUTOSIZE);
    
    // Display image
    imshow("Display window", image);
    waitKey(0);
    
    // Alocate buffer to hold each pixel RGBA values.
    unsigned char* pixels = (unsigned char*) malloc(sizeof(unsigned char) * image.cols * image.rows * 4);
    
    // Alocate buffer which will receive blur values.
    unsigned char* bluredPixels = (unsigned char*) malloc(sizeof(unsigned char) * image.cols * image.rows * 4);
    
    // Copy pixel data from image to buffer.
    int pixelIdx = 0;
    uchar* pixel;
    // Foreach row
    for(int i = 0; i < imageHeight; i++){
        pixel = image.ptr<uchar>(i);
        // Foreach column, assuming 3 color channels.
        for(int j = 0; j < imageWidth; j+=3){
            pixels[pixelIdx] = pixel[j]; // Blue
            pixels[pixelIdx + 1] = pixel[j+1]; // Green
            pixels[pixelIdx + 2] = pixel[j+2]; // Red
            pixels[pixelIdx + 3] = 0xff; // Alpha
            pixelIdx += 4; // Move to the next pixel.
        }
    }
    
    // First, try to obtain a dispatch queue that can send work to the
    // GPU in our system.
    dispatch_queue_t queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_GPU, NULL);
    
    // In the event that our system does NOT have an OpenCL-compatible GPU,
    // we can use the OpenCL CPU compute device instead.
    if (queue == NULL) {
        queue = gcl_create_dispatch_queue(CL_DEVICE_TYPE_CPU, NULL);
    }
    
    // Create OpenCL Image object.
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_UNSIGNED_INT8;
    
    // Image which will receive original image pixel values.
    cl_image dInputImage = gcl_create_image(&format,
                                            image.cols, // Image width
                                            image.rows, // Image height
                                            1, // this is a 2 dim image
                                            NULL);
    
    
    // Image which will receive blur values.
    cl_image dOutputImage = gcl_create_image(&format,
                                             image.cols, // Image width
                                             image.rows, // Image height
                                             1, // this is a 2 dim image
                                             NULL);
    
    // Make sure we've manage to create OpenCL image objects.
    if (!dInputImage || !dOutputImage)
    {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }
    
    const size_t cols = image.cols;
    const size_t rows = image.rows;
    
    // Dispatch the kernel block using one of the dispatch_ commands and the
    // queue created earlier.
    dispatch_sync(queue, ^{
        // This kernel is written such that each work item processes one pixel.
        // Thus, it executes over an OpenCL image object, with the width and
        // height of the image determining the dimensions
        // of execution.
        cl_ndrange range = {
            2,                      // The number of dimensions to use.
            {0, 0, 0},              // Start at the beginning of the image.
            {cols, rows, 0},   // Execute width * height work items.
            {0, 0, 0}   // Let OpenCL decide how to divide the work items into work-groups.
        };
        
        // Copy the host-side, initial pixel data to the image memory object on
        // the OpenCL device.
        const size_t origin[3] = { 0, 0, 0 };
        const size_t region[3] = { cols, rows, 1 };
        gcl_copy_ptr_to_image(dInputImage, pixels, origin, region);
        
        // Calling the kernel.
        blur_kernel(&range, dInputImage, dOutputImage);
        
        // Read back the results.
        gcl_copy_image_to_ptr(bluredPixels, dOutputImage, origin, region);
        
    });
    
    // Set blur values to image.
    pixelIdx = 0;
    // Foreach row
    for(int i = 0; i < imageHeight; i++){
        pixel = image.ptr<uchar>(i);
        // Foreach column
        for(int j = 0; j < imageWidth; j+=3){
            pixel[j] = bluredPixels[pixelIdx];
            pixel[j+1] = bluredPixels[pixelIdx+1];
            pixel[j+2] = bluredPixels[pixelIdx+2];
            pixelIdx += 4; // Move to the next pixel.
        }
    }
    
    // Display image
    imshow("Display window", image);
    waitKey(0);
    
    // Shutdown and cleanup
    // Don't forget to free up the CL device's memory when you're done.
    clReleaseMemObject(dInputImage);
    clReleaseMemObject(dOutputImage);
    
    // Finally, release your queue just as you would any GCD queue.
    dispatch_release(queue);
    
    // Clean up host-side allocations.
    free(pixels);
    free(bluredPixels);
    return 0;
}