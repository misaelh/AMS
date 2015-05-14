#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

/* Utility function/macro, used to do error checking.
Use this function/macro like this:
checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));
And to check the result of a kernel invocation:
checkCudaCall(cudaGetLastError());
*/
#define checkCudaCall(result) {                                     \
	if (result != cudaSuccess){                                     \
	cerr << "cuda error: " << cudaGetErrorString(result);       \
	cerr << " in " << __FILE__ << " at line "<< __LINE__<<endl; \
	exit(1);                                                    \
	}                                                               \
}

__global__ void rgb2grayCudaKernel(unsigned char *deviceImage, unsigned char *deviceResult, const int height, const int width){
    /* calculate the global thread id*/
    int threadsPerBlock  = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

    int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;
    int i = globalThreadNum;

    float grayPix = 0.0f;
    float r = static_cast< float >(deviceImage[i]);
    float g = static_cast< float >(deviceImage[(width * height) + i]);
    float b = static_cast< float >(deviceImage[(2 * width * height) + i]);
    grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

    deviceResult[i] = static_cast< unsigned char > (grayPix);
}

void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height) {

    unsigned char *deviceImage;
    unsigned char *deviceResult;

    int initialBytes = width * height * 3;  
    int endBytes =  width * height * sizeof(unsigned char);

    cudaMalloc((void**) &deviceImage, initialBytes);
    cudaMalloc((void**) &deviceResult, endBytes);
    cudaMemset(deviceResult, 0, endBytes);
    cudaMemset(deviceImage, 0, initialBytes);

    cudaError_t err = cudaMemcpy(deviceImage, inputImage, initialBytes, cudaMemcpyHostToDevice);    

    // Convert the input image to grayscale 
    rgb2grayCudaKernel<<<width * height / 256, 256>>>(deviceImage, deviceResult, height, width);
    cudaDeviceSynchronize();

    cudaMemcpy(grayImage, deviceResult, endBytes, cudaMemcpyDeviceToHost);

    ////// Sequential
    for ( int y = 0; y < height; y++ ) {
             for ( int x = 0; x < width; x++ ) {
                   float grayPix = 0.0f;
                   float r = static_cast< float >(inputImage[(y * width) + x]);
                   float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
                   float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

                   grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);
                   grayImage[(y * width) + x] = static_cast< unsigned char > (grayPix);
              }
        }

    //compare sequential and cuda and print pixels that are wrong
    for (int i = 0; i < endBytes; i++)
    {
        if (grayImage[i] != grayImage[i])
        cout << i << "-" << static_cast< unsigned int >(grayImage[i]) <<
                 " should be " << static_cast< unsigned int >(grayImage[i]) << endl;
        }

    cudaFree(deviceImage);
    cudaFree(deviceResult);
}

void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height) 
{
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for ( int x = 0; x < width; x++ ) 
		{
			float grayPix = 0.0f;
			float r = static_cast< float >(inputImage[(y * width) + x]);
			float g = static_cast< float >(inputImage[(width * height) + (y * width) + x]);
			float b = static_cast< float >(inputImage[(2 * width * height) + (y * width) + x]);

			grayPix = (0.3f * r) + (0.59f * g) + (0.11f * b);

			grayImage[(y * width) + x] = static_cast< unsigned char >(grayPix);
		}
	}
	// /Kernel
	kernelTime.stop();

	cout << fixed << setprecision(6);
	cout << "rgb2gray (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

/////////////////////////////////////
/*
__global__ void histogram1DCudaKernel
{
}
*/ 
__global__ void histogram1DCudaKernel(unsigned char *grayImg, unsigned int *hist, const int no_of_bins, const int width, const int height){
    /* calculate the global thread id*/
    int threadsPerBlock  = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

    int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	int startIdxHist = globalThreadNum*no_of_bins;
	int startIdxImg = globalThreadNum*width;

	for(int i = startIdxImg; i < startIdxImg + width && i<width*height; i++) {
		hist[startIdxHist+grayImg[i]]++;
	}
}

__global__ void sumHistCuda(unsigned int *histArray, unsigned int *hist, const int no_of_bins, const int height, const int width){
    /* calculate the global thread id*/
    int threadsPerBlock  = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

    int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;

	for(int i = 0; i < height; i++) {
		hist[globalThreadNum] += histArray[i*no_of_bins+globalThreadNum];
	}
}

void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage,const int width, const int height, 
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
	const unsigned int BAR_WIDTH)
{
	unsigned int max = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	unsigned int *histArray;
	unsigned int hist[256] = {0};
	histArray = (unsigned int*)malloc(height*HISTOGRAM_SIZE*sizeof(unsigned int));
	memset(histArray, 0, height*HISTOGRAM_SIZE*sizeof(unsigned int));

	unsigned char *grayImgCuda;
	unsigned int *histArrayComputedCuda;
	unsigned int *histCuda;

	cudaMalloc((void **) &histArrayComputedCuda, height*HISTOGRAM_SIZE*sizeof(unsigned int));
	cudaMemset(histArrayComputedCuda, 0, height*HISTOGRAM_SIZE*sizeof(unsigned int));
	cudaMalloc((void **) &grayImgCuda, width*height*sizeof(unsigned char));

	dim3 gridSize(16,1,1);
	dim3 blockSize(16,16,1);
	cudaMemcpy(grayImgCuda,grayImage,sizeof(unsigned char)*height*width,cudaMemcpyHostToDevice);
	histogram1DCudaKernel<<<gridSize, blockSize>>>(grayImgCuda, histArrayComputedCuda, HISTOGRAM_SIZE, width, height);
	cudaError err = cudaDeviceSynchronize();
	err = cudaMemcpy(histArray, histArrayComputedCuda, height*HISTOGRAM_SIZE*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	err = cudaMalloc((void **)&histCuda,HISTOGRAM_SIZE*sizeof(unsigned int));
	err = cudaMemset(histCuda, 0, HISTOGRAM_SIZE*sizeof(unsigned int));

	dim3 gridSize2(16,1,1);
	dim3 blockSize2(16,1,1);
	//err = cudaMemcpy(histArrayComputedCuda, histArray, height*HISTOGRAM_SIZE*sizeof(unsigned int), cudaMemcpyHostToDevice);
	sumHistCuda<<<gridSize, blockSize2>>>(histArrayComputedCuda, histCuda, 256, height, width);
	err = cudaDeviceSynchronize();
	err = cudaMemcpy(histogram, histCuda, HISTOGRAM_SIZE*sizeof(unsigned int), cudaMemcpyDeviceToHost);

	for ( unsigned int i = 0; i < HISTOGRAM_SIZE; i++ ) 
	{
		if ( histogram[i] > max ) 
		{
			max = histogram[i];
		}
	}

	for ( int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) 
	{
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for ( unsigned int y = 0; y < value; y++ ) 
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) 
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for ( unsigned int y = value; y < HISTOGRAM_SIZE; y++ ) 
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) 
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}

	cout << fixed << setprecision(6);
	cout << "histogram1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
	cudaFree(grayImgCuda);
	cudaFree(histArrayComputedCuda);
	cudaFree(histCuda);
}

void histogram1D(unsigned char *grayImage, unsigned char *histogramImage,const int width, const int height, 
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
	const unsigned int BAR_WIDTH) 
{
	unsigned int max = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	memset(reinterpret_cast< void * >(histogram), 0, HISTOGRAM_SIZE * sizeof(unsigned int));

	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for ( int x = 0; x < width; x++ ) 
		{
			histogram[static_cast< unsigned int >(grayImage[(y * width) + x])] += 1;
		}
	}
	// /Kernel
	kernelTime.stop();

	for ( unsigned int i = 0; i < HISTOGRAM_SIZE; i++ ) 
	{
		if ( histogram[i] > max ) 
		{
			max = histogram[i];
		}
	}

	for ( int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) 
	{
		unsigned int value = HISTOGRAM_SIZE - ((histogram[x / BAR_WIDTH] * HISTOGRAM_SIZE) / max);

		for ( unsigned int y = 0; y < value; y++ ) 
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) 
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 0;
			}
		}
		for ( unsigned int y = value; y < HISTOGRAM_SIZE; y++ ) 
		{
			for ( unsigned int i = 0; i < BAR_WIDTH; i++ ) 
			{
				histogramImage[(y * HISTOGRAM_SIZE * BAR_WIDTH) + x + i] = 255;
			}
		}
	}

	cout << fixed << setprecision(6);
	cout << "histogram1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

/////////////////////////////////////
/*
__global__ void contrast1DKernel
{
}
*/

/*
void contrast1DCuda
{
}
*/

void contrast1D(unsigned char *grayImage, const int width, const int height, 
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
	const unsigned int CONTRAST_THRESHOLD) 
{
	unsigned int i = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	while ( (i < HISTOGRAM_SIZE) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i++;
	}
	unsigned int min = i;

	i = HISTOGRAM_SIZE - 1;
	while ( (i > min) && (histogram[i] < CONTRAST_THRESHOLD) ) 
	{
		i--;
	}
	unsigned int max = i;
	float diff = max - min;

	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for (int x = 0; x < width; x++ ) 
		{
			unsigned char pixel = grayImage[(y * width) + x];

			if ( pixel < min ) 
			{
				pixel = 0;
			}
			else if ( pixel > max ) 
			{
				pixel = 255;
			}
			else 
			{
				pixel = static_cast< unsigned char >(255.0f * (pixel - min) / diff);
			}

			grayImage[(y * width) + x] = pixel;
		}
	}
	// /Kernel
	kernelTime.stop();

	cout << fixed << setprecision(6);
	cout << "contrast1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

/////////////////////////////////////
/*
__global__ void triangularSmoothKernel
{
}
*/

/*
void triangularSmoothCuda
{
}
*/

void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height,
	const float *filter) 
{
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	kernelTime.start();
	// Kernel
	for ( int y = 0; y < height; y++ ) 
	{
		for ( int x = 0; x < width; x++ ) 
		{
			unsigned int filterItem = 0;
			float filterSum = 0.0f;
			float smoothPix = 0.0f;

			for ( int fy = y - 2; fy < y + 3; fy++ ) 
			{
				for ( int fx = x - 2; fx < x + 3; fx++ ) 
				{
					if ( ((fy < 0) || (fy >= height)) || ((fx < 0) || (fx >= width)) ) 
					{
						filterItem++;
						continue;
					}

					smoothPix += grayImage[(fy * width) + fx] * filter[filterItem];
					filterSum += filter[filterItem];
					filterItem++;
				}
			}

			smoothPix /= filterSum;
			smoothImage[(y * width) + x] = static_cast< unsigned char >(smoothPix);
		}
	}
	// /Kernel
	kernelTime.stop();

	cout << fixed << setprecision(6);
	cout << "triangularSmooth (cpu): \t" << kernelTime.getElapsed() << " seconds." << endl;
}


