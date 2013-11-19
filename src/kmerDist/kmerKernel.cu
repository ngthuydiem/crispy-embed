/***********************************************
* # Copyright 2011. Thuy Diem Nguyen & Zejun Zheng
* # Contact: thuy1@e.ntu.edu.sg or zheng_zejun@sics.a-star.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

// Note: don't use_fast_math option
#include "kmerKernel.h"

texture<ushort, 2, cudaReadModeElementType> texRef;

texture<ushort, 2, cudaReadModeElementType> &getTexRef(void)
{
        return texRef;
}

__global__ void kmerKernel(int * pairArray, float *distArray, int maxNumTuples, int numPairs) 
{
	ushort matches = 0;
	ushort x, y, k, l;
	ushort tuple1Length, tuple2Length;
	int row, col;

	int pairIndex = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (pairIndex < numPairs) {
		row = pairArray[pairIndex*2];
		col = pairArray[pairIndex*2+1];
		
		tuple1Length = tex2D(texRef, maxNumTuples * (row & 63) + (maxNumTuples-1), row/64); // tex2D( texRef, width, height )
		tuple2Length = tex2D(texRef, maxNumTuples * (col & 63) + (maxNumTuples-1), col/64); // tex2D( texRef, width, height )

		for (k = 0, l = 0; (k < tuple1Length) && (l < tuple2Length);)
		{
			x = tex2D(texRef, maxNumTuples * (row & 63) + k, row/64);
			y = tex2D(texRef, maxNumTuples * (col & 63) + l, col/64);

			matches = matches + (ushort)(x==y);
			k = k + (ushort)(x<=y);
			l = l + (ushort)(x>=y);	
		}
		distArray[pairIndex] = 1.0f - (float) matches / min(tuple1Length, tuple2Length);		
	}
	else
		distArray[pairIndex] = -1.0f;
}

	

void launchKmerKernel(cudaStream_t stream, dim3 blocksPerGrid, dim3 threadsPerBlock, int * d_pairArray, float* d_distArray, int maxNumTuples, int numPairs) 
{	
	kmerKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_pairArray, d_distArray, maxNumTuples, numPairs);			
								
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
