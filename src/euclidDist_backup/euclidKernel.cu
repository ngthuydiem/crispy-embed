/***********************************************
* # Copyright 2011. Thuy Diem Nguyen & Zejun Zheng
* # Contact: thuy1@e.ntu.edu.sg or zheng_zejun@sics.a-star.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

// Note: don't use_fast_math option
#include "euclidKernel.h"

texture<float, 2, cudaReadModeElementType> texRef;

texture<float, 2, cudaReadModeElementType> &getTexRef(void)
{
        return texRef;
}

__global__ void euclidKernel(float *distArray, int numReads, int numSeeds, int stageX, int stageY, int arrayDim) 
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
			x = tex2D(texRef, numSeeds * (row & 15) + k, row/16);
			y = tex2D(texRef, numSeeds * (col & 15) + k, col/16);
			diff = x - y;
			sum += diff * diff;	
		}
		// convert from step size of 0.005 to 0.01			
		distArray[index] = sqrt(sum/numSeeds); 
	}
	else
		distArray[index] = 1.1f;
}

void launchEuclidKernel(cudaStream_t stream, dim3 blocksPerGrid, dim3 threadsPerBlock, float* d_distArray, int numReads, int numSeeds, int stageX, int stageY, int arrayDim) 
{	

	euclidKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_distArray, numReads, numSeeds, stageX, stageY, arrayDim);				
	
}


// General GPU Device CUDA Initialization
int gpuDeviceInit(int devID)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "gpuDeviceInit() CUDA error: no devices supporting CUDA.\n");
        exit(-1);
    }
    if (devID < 0) 
        devID = 0;
    if (devID > deviceCount-1) {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuDeviceInit (-device=%d) is not a valid GPU device. <<\n", devID);
        fprintf(stderr, "\n");
        return -devID;
    }
    
    cudaDeviceProp deviceProp;
    checkCudaErrors( cudaGetDeviceProperties(&deviceProp, devID) );
    if (deviceProp.major < 1) {
        fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
        exit(-1);                                                  \
    }
    
    checkCudaErrors( cudaSetDevice(devID) );
    printf("> gpuDeviceInit() CUDA device [%d]: %s\n", devID, deviceProp.name);
    return devID;
}
