// Simple starting example for CUDA program
// Kees Lemmens, last change May 2012

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define NRBLKS  4 // Nr of blocks in a kernel (gridDim)
#define NRTPBK  4 // Nr of threads in a block (blockDim)

void checkCudaError(char *error)
{
   if (cudaGetLastError() != cudaSuccess)
   {
      fprintf (stderr, "Cuda : %s\n",error);
      exit(EXIT_FAILURE);
   }
}

__global__ void decodeOnGPU(char *string)
{
   // Each thread decodes a single character from the encoded string
   int myid = (blockIdx.x * blockDim.x) + threadIdx.x;
   char decoder =(char)011; // 000001001 = 011 in octal
   string[myid] ^= decoder;
}

int main(void)
{
   char *encryption = "Aleef)^f{em)(((\11";
   
   char *string_h; // pointer to host   memory
   char *string_d; // pointer to device memory
   int len = NRTPBK * NRBLKS;
   
   // allocate memory on device
   string_h = (char *)malloc(len);
   cudaMalloc((void **) &string_d, len);
   checkCudaError("Malloc failed on GPU device !");
   
   // copy encrypted string to device memory
   cudaMemcpy(string_d, encryption, sizeof(char) * len, cudaMemcpyHostToDevice);
   checkCudaError("Sending data to GPU device failed !");

   // Start the decoding GPU kernel on NRTPBK threads and NRBLKS blocks
   decodeOnGPU <<< NRBLKS, NRTPBK >>> (string_d);
   checkCudaError("Kernel failed on GPU device !");

   // retrieve data from GPU device: string_d to string_h
   cudaMemcpy(string_h, string_d, len, cudaMemcpyDeviceToHost);
   checkCudaError("Receiving data from GPU device failed !");
   
   printf("%s \n", string_h);
   
   // cleanup memory
   cudaFree(string_d);
   free(string_h);
}
