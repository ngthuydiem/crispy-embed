/***********************************************
* # Copyright 2011. Thuy Diem Nguyen & Zejun Zheng
* # Contact: thuy1@e.ntu.edu.sg or zheng_zejun@sics.a-star.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

// Note: don't use_fast_math option
#include "euclidKernel.h"

__global__ void euclidKernel(float *in, float *out, int numReads, int numSeeds, int stageX, int stageY, int arrayDim) 
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;	
	int j = blockIdx.x * blockDim.x + threadIdx.x;	

	int row = stageX * arrayDim + i;
	int col = stageY * arrayDim + j;
	int index = i * arrayDim + j;
	
	float x, y, diff, sum;
	int k;

	if ( (row < col) && (col < numReads) )
	{			
		sum = 0;
		for (k = 0; k < numSeeds; ++k) {
			x = in[numSeeds * row + k];
			y = in[numSeeds * col + k];
			diff = x - y;
			sum += diff * diff;	
		}
		// convert from step size of 0.005 to 0.01			
		out[index] = sqrt(sum/numSeeds); 
		//printf("%d %d %.3f\n", row, col, out[index]);

	}
	else {
		out[index] = 1.1f;
		
	}
}

void launchEuclidKernel(cudaStream_t stream, dim3 blocksPerGrid, dim3 threadsPerBlock, float * d_in, float* d_out, int numReads, int numSeeds, int stageX, int stageY, int arrayDim) 
{	

	euclidKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_in, d_out, numReads, numSeeds, stageX, stageY, arrayDim);				

}


// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        printf("gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }
    if (devID < 0) 
        devID = 0;
    if (devID > deviceCount-1) {
        printf("\n");
        printf(">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        printf(">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        printf("\n");
        return -devID;
    }
    
    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
    if (deviceProp.major < 1) {
        printf("gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(-1);                                                  \
    }
    
    checkCudaErrors( cudaSetDevice(devID) );
    printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
    return devID;
}
