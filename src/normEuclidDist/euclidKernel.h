/***********************************************
* # Copyright 2011. Thuy Diem Nguyen
* # Contact: thuy1@e.ntu.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

#ifndef _EUCLID_KERNEL_H_
#define _EUCLID_KERNEL_H_

#include "euclidMain.h"
////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg)      __getLastCudaError (msg, __FILE__, __LINE__)
inline void __getLastCudaError( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}


//void launchEuclidKernel(cudaStream_t stream, dim3 blocksPerGrid, dim3 threadsPerBlock, float *in, float* out, int numReads, int numSeeds, int stageX, int stageY, int arrayDim, float * variance);
void launchEuclidKernel(cudaStream_t stream, dim3 blocksPerGrid, dim3 threadsPerBlock, float *in, float* out, int numReads, int numSeeds, int stageX, int stageY, int arrayDim, float * variance);

int gpuDeviceInit(int devID);

#endif
