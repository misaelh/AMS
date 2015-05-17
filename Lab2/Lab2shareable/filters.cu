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

__constant__ float filter_sum = 35.0f;

__constant__ float filterCuda[] = {	1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
						1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
						1.0f, 1.0f, 1.0f, 1.0f, 1.0f
};
__global__ void rgb2grayCudaKernel(unsigned char *deviceImage, unsigned char *deviceResult, const int height, const int width){
	/* calculate the global thread id*/
	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;
	int i = globalThreadNum;

	int grayPix = 0;
	grayPix = (30*deviceImage[i] + 59 * deviceImage[(width * height) + i] + 11 * deviceImage[(2 * width * height) + i])/100;
	deviceResult[i] = grayPix;
}

void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height) {

	unsigned char *deviceImage;
	unsigned char *deviceResult;
	unsigned int imageSize = width * height;

	int initialBytes = width * height * 3 *sizeof(unsigned char);
	int endBytes =  width * height * sizeof(unsigned char);
	unsigned int xGridDim = 0, yGridDim=1;

	cudaError_t err = cudaMalloc((void**) &deviceImage, initialBytes);
	err = cudaMalloc((void**) &deviceResult, endBytes);
	err = cudaMemset(deviceResult, 0, endBytes);
	err = cudaMemset(deviceImage, 0, initialBytes);

	err = cudaMemcpy(deviceImage, inputImage, initialBytes, cudaMemcpyHostToDevice);
	if(imageSize >= 8192*8192)
		if(imageSize%(8192*8192-1)==0)
			yGridDim = imageSize / (8192*8192-1);
		else
			yGridDim = imageSize / (8192*8192-1)+ 1;

	if(imageSize%1024==0)
		xGridDim = width*height / (1024*yGridDim);
	else
		xGridDim = width*height / (1024*yGridDim) + 1;

	// Convert the input image to grayscale 
	dim3 grid(xGridDim,yGridDim,1);
	dim3 block(32,32,1);

	rgb2grayCudaKernel<<<grid, block>>>(deviceImage, deviceResult, height, width);
	err = cudaDeviceSynchronize();

	err = cudaMemcpy(grayImage, deviceResult, endBytes, cudaMemcpyDeviceToHost);
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

	//cout << fixed << setprecision(6);
	//cout << "rgb2gray (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

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
	unsigned int xGridDim = 0;

	cudaMalloc((void **) &histArrayComputedCuda, height*HISTOGRAM_SIZE*sizeof(unsigned int));
	cudaMemset(histArrayComputedCuda, 0, height*HISTOGRAM_SIZE*sizeof(unsigned int));
	cudaMalloc((void **) &grayImgCuda, width*height*sizeof(unsigned char));

	if(height%1024==0)
		xGridDim = height / 1024;
	else
		xGridDim = height / 1024 + 1;

	dim3 gridSize(xGridDim,1,1);
	dim3 blockSize(32,32,1);
	cudaMemcpy(grayImgCuda,grayImage,sizeof(unsigned char)*height*width,cudaMemcpyHostToDevice);
	histogram1DCudaKernel<<<gridSize, blockSize>>>(grayImgCuda, histArrayComputedCuda, HISTOGRAM_SIZE, width, height);
	cudaError err = cudaDeviceSynchronize();
	err = cudaMemcpy(histArray, histArrayComputedCuda, height*HISTOGRAM_SIZE*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	err = cudaMalloc((void **)&histCuda,HISTOGRAM_SIZE*sizeof(unsigned int));
	err = cudaMemset(histCuda, 0, HISTOGRAM_SIZE*sizeof(unsigned int));

	dim3 gridSize2(1,1,1);
	dim3 blockSize2(16,16,1);
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

	for ( unsigned int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) 
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

	//cout << fixed << setprecision(6);
	//cout << "histogram1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
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

	for ( unsigned int x = 0; x < HISTOGRAM_SIZE * BAR_WIDTH; x += BAR_WIDTH ) 
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

	//cout << fixed << setprecision(6);
	//cout << "histogram1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

__global__ void contrast1DCudaKernel(unsigned char *deviceImage, unsigned char *deviceResult, const int height, const int width,
								 unsigned int min, unsigned int max, float diff)
{
	int threadsPerBlock  = blockDim.x * blockDim.y;
    int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
    int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

    int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;
    int i = globalThreadNum;

unsigned int grayPix = static_cast< unsigned int >(deviceImage[i]);

	if ( grayPix < min ) 
	{
		grayPix = 0;
	}
	else if ( grayPix > max ) 
	{
		grayPix = 255;
	}
	else 
	{
		grayPix = (255 * (grayPix - min) / diff);
	}

	deviceResult[i] = static_cast< unsigned char > (grayPix);
}

void contrast1DCuda(unsigned char *grayImage, const int width, const int height, 
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
	const unsigned int CONTRAST_THRESHOLD) 
{
    unsigned char *deviceImage;
    unsigned char *deviceResult;
	unsigned int imageSize = width * height;

    int numBytes =  width * height * sizeof(unsigned char);
	unsigned int i = 0, xGridDim = 0, yGridDim = 1;
	unsigned int maxHist = 0;

	for ( unsigned int i = 0; i < HISTOGRAM_SIZE; i++ ) 
	{
		if ( histogram[i] > maxHist ) 
		{
			maxHist = histogram[i];
		}
	}

	i=0;
	while ( (i < HISTOGRAM_SIZE) && ((histogram[i]*HISTOGRAM_SIZE)/maxHist < CONTRAST_THRESHOLD) ) 
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

	cudaMalloc((void**) &deviceImage, numBytes);
    cudaMalloc((void**) &deviceResult, numBytes);
    cudaMemset(deviceResult, 0, numBytes);
    cudaMemset(deviceImage, 0, numBytes);

    cudaError_t err = cudaMemcpy(deviceImage, grayImage, numBytes, cudaMemcpyHostToDevice);    
	if(imageSize >= 8192*8192)
		if(imageSize%(8192*8192-1)==0)
			yGridDim = imageSize / (8192*8192-1);
		else
			yGridDim = imageSize / (8192*8192-1)+ 1;

	if(imageSize%1024==0)
		xGridDim = width*height / (1024*yGridDim);
	else
		xGridDim = width*height / (1024*yGridDim) + 1;

	// Convert the input image to grayscale 
	dim3 grid(xGridDim,yGridDim,1);
	dim3 block(32,32,1);
    // Convert the input image to grayscale 
    contrast1DCudaKernel<<<grid, block>>>(deviceImage, deviceResult, height, width, min, max, diff);
    cudaDeviceSynchronize();

    cudaMemcpy(grayImage, deviceResult, numBytes, cudaMemcpyDeviceToHost);

}

void contrast1D(unsigned char *grayImage, const int width, const int height, 
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
	const unsigned int CONTRAST_THRESHOLD) 
{
	unsigned int i = 0;
	unsigned int maxHist = 0;
	NSTimer kernelTime = NSTimer("kernelTime", false, false);

	for ( unsigned int j = 0; j < HISTOGRAM_SIZE; j++ ) 
	{
		if ( histogram[j] > maxHist ) 
		{
			maxHist = histogram[j];
		}
	}

	while ( (i < HISTOGRAM_SIZE) && ((histogram[i]*HISTOGRAM_SIZE/maxHist) < CONTRAST_THRESHOLD) ) 
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

	//cout << fixed << setprecision(6);
	//cout << "contrast1D (cpu): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
}

__global__ void triangularSmoothKernel(unsigned char *grayScale, unsigned char *smoothened, unsigned int width, unsigned int height)
{
	int threadsPerBlock  = blockDim.x * blockDim.y;
	int threadNumInBlock = threadIdx.x + blockDim.x * threadIdx.y;
	int blockNumInGrid   = blockIdx.x  + gridDim.x  * blockIdx.y;

	int globalThreadNum = blockNumInGrid * threadsPerBlock + threadNumInBlock;
	int pixelPos = globalThreadNum;
	int modWidth = pixelPos%width;
	int modHeight = (pixelPos/width);

	int x, y;
	float smoothened_0 = 0, smoothened_1 = 0, smoothened_2 = 0, smoothened_3 = 0, smoothened_4 = 0, smoothened_f = 0;
	bool unrollFlag = true;
	int width2pos = 2*width, width2minus = -width2pos;

	int x_start = 0, x_end = 5, y_start = 0, y_end = 5;

	if(pixelPos >= width * height)
		return;

	if((modWidth < 2) || (modWidth > width - 3)) {
		unrollFlag = false;
		if(modWidth < 2)
			x_start = 2 - modWidth;
		else
			x_end = 2 + width - modWidth;
	}

	if((modHeight < 2) || (modHeight > height - 3)) {
		unrollFlag = false;
		if(modHeight < 2)
			y_start = 2 - modHeight;
		else
			y_end = 2 + height - modHeight;
	}

	if(!unrollFlag){
		float el_sum = 0;
		for(y = y_start; y < y_end; y++){
			for(x = x_start; x < x_end; x++) {
				smoothened_f += filterCuda[5*y+x] * grayScale[pixelPos+x-2+(y-2)*width];
				el_sum += filterCuda[5*y+x];
			}
		}
		smoothened_f/=el_sum;
		smoothened[pixelPos] = smoothened_f;
	}
	else {
		smoothened_0 += filterCuda[0] * grayScale[pixelPos-2+width2minus];
		smoothened_1 += filterCuda[1] * grayScale[pixelPos-1+width2minus];
		smoothened_2 += filterCuda[2] * grayScale[pixelPos+0+width2minus];
		smoothened_3 += filterCuda[3] * grayScale[pixelPos+1+width2minus];
		smoothened_4 += filterCuda[4] * grayScale[pixelPos+2+width2minus];

		smoothened_0 += filterCuda[5] * grayScale[pixelPos-2-width];
		smoothened_1 += filterCuda[6] * grayScale[pixelPos-1-width];
		smoothened_2 += filterCuda[7] * grayScale[pixelPos+0-width];
		smoothened_3 += filterCuda[8] * grayScale[pixelPos+1-width];
		smoothened_4 += filterCuda[9] * grayScale[pixelPos+2-width];

		smoothened_0 += filterCuda[10] * grayScale[pixelPos-2];
		smoothened_1 += filterCuda[11] * grayScale[pixelPos-1];
		smoothened_2 += filterCuda[12] * grayScale[pixelPos+0];
		smoothened_3 += filterCuda[13] * grayScale[pixelPos+1];
		smoothened_4 += filterCuda[14] * grayScale[pixelPos+2];

		smoothened_0 += filterCuda[15] * grayScale[pixelPos-2+width];
		smoothened_1 += filterCuda[16] * grayScale[pixelPos-1+width];
		smoothened_2 += filterCuda[17] * grayScale[pixelPos+0+width];
		smoothened_3 += filterCuda[18] * grayScale[pixelPos+1+width];
		smoothened_4 += filterCuda[19] * grayScale[pixelPos+2+width];

		smoothened_0 += filterCuda[20] * grayScale[pixelPos-2+width2pos];
		smoothened_1 += filterCuda[21] * grayScale[pixelPos-1+width2pos];
		smoothened_2 += filterCuda[22] * grayScale[pixelPos+0+width2pos];
		smoothened_3 += filterCuda[23] * grayScale[pixelPos+1+width2pos];
		smoothened_4 += filterCuda[24] * grayScale[pixelPos+2+width2pos];
		
		smoothened_0 = smoothened_0 + smoothened_1;
		smoothened_3 = smoothened_3 + smoothened_4;
		smoothened_0 += smoothened_2 + smoothened_3;

		smoothened_0/=filter_sum;
		smoothened[pixelPos] = smoothened_0;
	}
}

void triangularSmoothCuda(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height)
{
	unsigned char *cudaImGray, *cudaEnhanced;
	unsigned int xGridDim = 0, yGridDim = 1;
	unsigned int imageSize = width * height;

	cudaMalloc((void**)&cudaImGray, height*width*sizeof(unsigned char));
	cudaMalloc((void**)&cudaEnhanced, height*width*sizeof(unsigned char));

	cudaMemcpy(cudaImGray, grayImage, height*width*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemset(cudaEnhanced, 0, height*width*sizeof(unsigned char));

	if(imageSize >= 8192*8192)
		if(imageSize%(8192*8192-1)==0)
			yGridDim = imageSize / (8192*8192-1);
		else
			yGridDim = imageSize / (8192*8192-1)+ 1;

	if(imageSize%1024==0)
		xGridDim = width*height / (1024*yGridDim);
	else
		xGridDim = width*height / (1024*yGridDim) + 1;

	// Convert the input image to grayscale 
	dim3 grid(xGridDim,yGridDim,1);
	dim3 block(32,32,1);

	triangularSmoothKernel<<<grid, block>>> (cudaImGray, cudaEnhanced, width, height);
	cudaError err = cudaMemcpy(smoothImage, cudaEnhanced ,height*width*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	cudaFree(cudaImGray);
	cudaFree(cudaEnhanced);
}

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

	//cout << fixed << setprecision(6);
	//cout << "triangularSmooth (cpu): \t" << kernelTime.getElapsed() << " seconds." << endl;
}