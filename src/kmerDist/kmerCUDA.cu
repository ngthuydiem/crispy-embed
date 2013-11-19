/***********************************************
* # Copyright 2011. Thuy Diem Nguyen & Zejun Zheng
* # Contact: thuy1@e.ntu.edu.sg or zheng_zejun@sics.a-star.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

// Note: don't use_fast_math option

#include "kmerMain.h"
#include "kmerKernel.h"

void computeKmerDist_CUDA(READ* &readArray, FILE* pairFile, FILE* distFile, int numReads, int maxLen, int K) {
	
	int i, j, row, length, EOFTag = 0;	
	long totalNumPairs = 0;
	int numPairs = 0;
			
	int maxNumTuples = maxLen - K + 1 + 1;
	
	if (maxNumTuples%16 != 0)
			maxNumTuples += 16 - (maxNumTuples%16);	
	printf("maxNumTuples: %d\n", maxNumTuples);
			
	// determine gridSize and blockSize
	size_t threadsPerBlock(BLOCK_SIZE);
	size_t blocksPerGrid(GRID_SIZE);
		
	// declare host variables
	ushort *tupleSet;	
	// allocate host memory
	tupleSet = (ushort *) malloc(numReads * maxNumTuples * sizeof(ushort));
	
	for (i = 0; i < numReads; ++i)
	{
		row = i * maxNumTuples;
		length = readArray[i].length - K + 1;
		tupleSet[row + maxNumTuples - 1] = length;
		for (j = 0; j < length; ++j)
			tupleSet[row + j] = readArray[i].tuples[j];	
	}

	for (i = 0; i < numReads; ++i)
		readArray[i].finalize();
	free(readArray);

	// Allocate space for the pair id array	
	int *h_pairArray;
	checkCudaErrors( cudaMallocHost((void**)&h_pairArray, NUM_PAIRS * 2 * sizeof(int)) );	
	int *d_pairArray;
	checkCudaErrors( cudaMalloc((void**)&d_pairArray, NUM_PAIRS * 2 * sizeof(int)) );	

	// Allocate memory block for the Needleman-Wunsch distance array	
	float *h_distArray;	
	checkCudaErrors( cudaMallocHost((void**)&h_distArray, NUM_PAIRS * sizeof(float)) );
		
	// declare device variables
	float *d_distArray;	
	// allocate device memory	
	checkCudaErrors( cudaMalloc((void**)&d_distArray, NUM_PAIRS * sizeof(float)) );	

	cudaStream_t stream[NUM_STREAMS];
	for (i = 0; i < NUM_STREAMS; ++i)
		checkCudaErrors( cudaStreamCreate(&stream[i]) );

	// use cudaArray to store tupleArraySet
	cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc<ushort>();
	cudaArray *cuArray;
	size_t width, height;
	width = maxNumTuples*64;
	height = numReads/64;
	if ( (numReads&63) != 0) 
	 	++height;
	checkCudaErrors( cudaMallocArray(&cuArray, &channelDesc, width, height) );
	checkCudaErrors( cudaMemcpyToArray(cuArray, 0, 0, tupleSet, maxNumTuples * numReads * sizeof(ushort), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaBindTextureToArray(getTexRef(), cuArray, channelDesc) );
	free(tupleSet);
	
	int offset, chunkSize, numPairsPerChunk;
	while (!EOFTag)
	{
		numPairs = loadPairs(pairFile, h_pairArray, EOFTag);	
		numPairsPerChunk = (numPairs + NUM_STREAMS - 1) / NUM_STREAMS;		
		if (numPairsPerChunk%16 != 0)
			numPairsPerChunk += 16 - (numPairsPerChunk%16);	

		for (i = 0; i < NUM_STREAMS; ++i) {
			offset = i * numPairsPerChunk;
			
			if (i < NUM_STREAMS - 1)
				chunkSize = numPairsPerChunk;
			else
				chunkSize = numPairs - offset;
			
			//printf("stream: %d, offset: %d, numPairs = %d, chunkSize: %d\n", i, offset, numPairs, chunkSize);
			
			checkCudaErrors( cudaMemcpyAsync(d_pairArray+offset*2, h_pairArray+offset*2, chunkSize * sizeof(int) * 2, cudaMemcpyHostToDevice, stream[i]) );
		
			// processing the kernel function
			launchKmerKernel(stream[i], blocksPerGrid, threadsPerBlock, d_pairArray+offset*2, d_distArray+offset,  maxNumTuples, chunkSize); 	
	
			// copy results from device to host
			checkCudaErrors( cudaMemcpyAsync(h_distArray+offset, d_distArray+offset, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream[i]) );	
		}
		
		for (i = 0; i < NUM_STREAMS; ++i) {						
			offset = i * numPairsPerChunk;
			
			if (i < NUM_STREAMS - 1)
				chunkSize = numPairsPerChunk;
			else
				chunkSize = numPairs - offset;
			
			checkCudaErrors( cudaStreamSynchronize(stream[i]) );
			
			writeToFile(distFile, h_distArray+offset, chunkSize, totalNumPairs);
		}
	}	

	for (i = 0; i < NUM_STREAMS; ++i)
		checkCudaErrors( cudaStreamDestroy(stream[i]) );
		
	// clean up host variables	
	checkCudaErrors( cudaFreeHost(h_pairArray) );
	checkCudaErrors( cudaFreeHost(h_distArray) );	
	
	checkCudaErrors( cudaFree(d_distArray) );
	
	// clean up device variables
	checkCudaErrors( cudaUnbindTexture(getTexRef()) );
	checkCudaErrors( cudaFreeArray(cuArray) );
	
	printf("totalNumPairs: %ld\n", totalNumPairs);	
}


