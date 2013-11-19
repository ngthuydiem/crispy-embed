/***********************************************
* # Copyright 2011. Thuy Diem Nguyen & Zejun Zheng
* # Contact: thuy1@e.ntu.edu.sg or zheng_zejun@sics.a-star.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

// Note: don't use_fast_math option
#include "euclidKernel.h"

__global__ void euclidKernel(float *inDistSet, float *distArray, int numReads, int numSeeds) 
{
	__shared__ float Ys[BLOCK_DIM][BLOCK_DIM];
	__shared__ float Xs[BLOCK_DIM][BLOCK_DIM];
	
	int bx = blockIdx.x, by = blockIdx.y;
	int tx = threadIdx.x, ty = threadIdx.y;		
	
	int yBegin = by*BLOCK_DIM*numSeeds;
	int xBegin = bx*BLOCK_DIM*numSeeds;	
	
	int yEnd = yBegin + numSeeds - 1;
	int y, x, k, index ;
	float diff, sum = 0;

	for (y = yBegin, x = xBegin; y < yEnd; y+=BLOCK_DIM, x+= BLOCK_DIM) {

		Ys[ty][tx] = inDistSet[y + ty * numSeeds + tx];
		Xs[tx][ty] = inDistSet[x + ty * numSeeds + tx];
		
		__syncthreads();
		
		for (k = 0; k < BLOCK_DIM; ++k) {
			diff = Ys[ty][k] - Xs[k][tx];
			sum += diff * diff;
		}
			 
		__syncthreads();
	}
	
	index = (by*BLOCK_DIM + ty) * numReads + bx*BLOCK_DIM + tx;
	distArray[index] = sqrtf(sum/numSeeds);
}

void launchEuclidKernel(dim3 blocksPerGrid, dim3 threadsPerBlock, float *d_inDistSet, float* d_distArray, int numReads, int numSeeds) 
{	

	euclidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_inDistSet, d_distArray, numReads, numSeeds);				
	
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
