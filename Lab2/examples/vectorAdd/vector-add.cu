#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "Timer.hpp"

using namespace std;
using LOFAR::NSTimer;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void vectorAddKernel(double* deviceA, double* deviceB, double* deviceResult) 
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
	
    deviceResult[index] = deviceA[index] + deviceB[index];
}


void vectorAddCuda(int n, double* a, double* b, double* result) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    double* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(double)));
    if (deviceA == NULL) {
        cout << "could not allocate memory!" << endl;
        return;
    }
    double* deviceB = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceB, n * sizeof(double)));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        cout << "could not allocate memory!" << endl;
        return;
    }
    double* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(double)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        cout << "could not allocate memory!" << endl;
        return;
    }

    NSTimer kernelTime = NSTimer("kernelTime", false, false);
    NSTimer memoryTime = NSTimer("memoryTime", false, false);

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(double), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(double), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime.start();
    vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceA, deviceB, deviceResult);
    cudaDeviceSynchronize();
    kernelTime.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(result, deviceResult, n * sizeof(double), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

    cout << fixed << setprecision(6);
    cout << "vector-add (kernel): \t\t" << kernelTime.getElapsed() << " seconds." << endl;
    cout << "vector-add (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;
}


int main(int argc, char* argv[]) {
    int n = 8 * 1024 * 1024;
    NSTimer vectorAddTimer("vector add timer");
    double* a = new double[n];
    double* b = new double[n];
    double* result = new double[n];
    double* resultCpu = new double[n];

    // initialize the vectors.
    for(int i=0; i<n; i++) {
        a[i] = i;
        b[i] = i;
    }

	for(int i=0; i<n; i++) {
		resultCpu[i] = a[i] + b[i] ;
	}

    vectorAddCuda(n, a, b, result);

    // verify the resuls
    for(int i=0; i<n; i++) {
        if( fabs(resultCpu[i] - result[i]) > 0.01  ) {
            cout	<< "error in results! Element " << i << " is " << result[i] 
					<< ", but should be " << resultCpu[i] << endl;
            exit(1);
        }
    }
    cout << "results OK!" << endl;
            
    delete[] a;
    delete[] b;
    delete[] result;
    
    return 0;
}
