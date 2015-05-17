//#define cimg_use_jpeg
#include <CImg.h>
#include <Timer.hpp>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <WinBase.h>

using cimg_library::CImg;
using LOFAR::NSTimer;
using std::cout;
using std::cerr;
using std::endl;
using std::fixed;
using std::setprecision;

// Constants
const bool displayImages = true;
const bool saveAllImages = true;
const unsigned int HISTOGRAM_SIZE = 256;
const unsigned int BAR_WIDTH = 1;
const unsigned int CONTRAST_THRESHOLD = 80;
const float filter[] = {	1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
						1.0f, 2.0f, 3.0f, 2.0f, 1.0f, 
						1.0f, 2.0f, 2.0f, 2.0f, 1.0f, 
						1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

extern void rgb2gray(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height);
extern void rgb2grayCuda(unsigned char *inputImage, unsigned char *grayImage, const int width, const int height);

extern void histogram1D(unsigned char *grayImage, unsigned char *histogramImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int BAR_WIDTH);
extern void histogram1DCuda(unsigned char *grayImage, unsigned char *histogramImage,const int width, const int height, 
	unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, 
	const unsigned int BAR_WIDTH);

extern void contrast1D(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD);
extern void contrast1DCuda(unsigned char *grayImage, const int width, const int height, unsigned int *histogram, const unsigned int HISTOGRAM_SIZE, const unsigned int CONTRAST_THRESHOLD);

extern void triangularSmooth(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height, const float *filter);
extern void triangularSmoothCuda(unsigned char *grayImage, unsigned char *smoothImage, const int width, const int height);

int main(int argc, char *argv[]) 
{
	if ( argc != 2 ) {
		cerr << "Usage: " << argv[0] << " <filename>" << endl;
		return 1;
	}
	LARGE_INTEGER count;
	double PCFreq = 0.0;
	__int64 start = 0;
	if(!QueryPerformanceFrequency(&count))
		cout << "QueryPerformanceFrequency failed!\n";
	PCFreq = double(count.QuadPart)/1000.0;

	// Load the input image
	CImg< unsigned char > inputImage = CImg< unsigned char >(argv[1]);
	if ( displayImages ) {
	//	inputImage.display("Input Image");
	}
	if ( inputImage.spectrum() != 3 ) {
		cerr << "The input must be a color image." << endl;
		return 1;
	}

	// Convert the input image to grayscale 
	CImg< unsigned char > grayImage = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);
	CImg< unsigned char > grayImageCuda = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);

	QueryPerformanceCounter(&count);
	start = count.QuadPart;
	rgb2gray(inputImage.data(), grayImage.data(), inputImage.width(), inputImage.height());
	QueryPerformanceCounter(&count);
	start = double(count.QuadPart - start)/PCFreq;

	QueryPerformanceCounter(&count);
	start = count.QuadPart;
	rgb2grayCuda(inputImage.data(), grayImageCuda.data(), inputImage.width(), inputImage.height());
	QueryPerformanceCounter(&count);
	start = double(count.QuadPart - start)/PCFreq;
	if ( displayImages ) {
		//grayImageCuda.display("Grayscale Image Cuda");
		//grayImage.display("Grayscale Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./grayscale.bmp");
	}
	// Compute 1D histogram
	CImg< unsigned char > histogramImage = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	CImg< unsigned char > histogramImageCuda = CImg< unsigned char >(BAR_WIDTH * HISTOGRAM_SIZE, HISTOGRAM_SIZE, 1, 1);
	unsigned int *histogram = new unsigned int [HISTOGRAM_SIZE];
	unsigned int *histogramCuda = new unsigned int[HISTOGRAM_SIZE];

	QueryPerformanceCounter(&count);
	start = count.QuadPart;
	histogram1D(grayImage.data(), histogramImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, BAR_WIDTH);
	QueryPerformanceCounter(&count);
	start = double(count.QuadPart - start)/PCFreq;
	printf("Histogram compute sequential: %d miliseconds\n",start);

	QueryPerformanceCounter(&count);
	start = count.QuadPart;
	histogram1DCuda(grayImageCuda.data(), histogramImageCuda.data(), grayImage.width(), grayImage.height(), histogramCuda, HISTOGRAM_SIZE, BAR_WIDTH);
	QueryPerformanceCounter(&count);
	start = double(count.QuadPart - start)/PCFreq;
	printf("Histogram compute CUDA: %d miliseconds\n",start);

	if ( displayImages ) {
		//histogramImage.display("Histogram");
		//histogramImageCuda.display("Histogram Cudas");
	}
	if ( saveAllImages ) {
		histogramImage.save("./histogram.bmp");
	}

	// Contrast enhancement
	QueryPerformanceCounter(&count);
	start = count.QuadPart;
	contrast1D(grayImage.data(), grayImage.width(), grayImage.height(), histogram, HISTOGRAM_SIZE, CONTRAST_THRESHOLD);
	QueryPerformanceCounter(&count);
	start = double(count.QuadPart - start)/PCFreq;
	printf("Constrast enhancement sequential: %d miliseconds\n",start);

	QueryPerformanceCounter(&count);
	start = count.QuadPart;
	contrast1DCuda(grayImageCuda.data(), grayImageCuda.width(), grayImageCuda.height(), histogramCuda, HISTOGRAM_SIZE, CONTRAST_THRESHOLD);
	QueryPerformanceCounter(&count);
	start = double(count.QuadPart - start)/PCFreq;
	printf("Constrast enhancement CUDA: %d miliseconds\n",start);

	if ( displayImages ) {
		//grayImage.display("Contrast Enhanced Image");
	}
	if ( saveAllImages ) {
		grayImage.save("./contrast.bmp");
	}

	delete [] histogram;

	// Triangular smooth (convolution)
	CImg< unsigned char > smoothImage = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);
	CImg< unsigned char > smoothImageCuda = CImg< unsigned char >(grayImage.width(), grayImage.height(), 1, 1);

	QueryPerformanceCounter(&count);
	start = count.QuadPart;
	triangularSmooth(grayImage.data(), smoothImage.data(), grayImage.width(), grayImage.height(), filter);
	QueryPerformanceCounter(&count);
	start = double(count.QuadPart - start)/PCFreq;
	printf("Smooth compute sequential: %d miliseconds\n",start);

	QueryPerformanceCounter(&count);
	start = count.QuadPart;
	triangularSmoothCuda(grayImageCuda.data(), smoothImageCuda.data(), grayImageCuda.width(), grayImageCuda.height());
	QueryPerformanceCounter(&count);
	start = double(count.QuadPart - start)/PCFreq;
	printf("Smooth compute CUDA: %d miliseconds\n",start);
	
	unsigned int cnt=0;
	CImg< unsigned char > grayImageDiff = CImg< unsigned char >(inputImage.width(), inputImage.height(), 1, 1);
	for ( int y = 0; y < grayImage.height()*grayImage.width(); y++ ) 
	{
		grayImageDiff[y] = smoothImage[y] - smoothImageCuda[y];
			if(grayImageDiff[y]!=0){
				cnt++;
				grayImageDiff[y] = 255;
			}
	}
	cout << "Count errors: " << cnt << endl;

	if ( displayImages ) {
		grayImageDiff.display("Smooth Image Diff");
		//smoothImage.display("Smooth Image");
		//smoothImageCuda.display("Smooth Image Cuda");
	}
	
	//if ( saveAllImages ) {
	//	smoothImage.save("./smooth.bmp");
	//	smoothImageCuda.save("./smoothCuda.bmp");
	//}

 	return 0;
}

