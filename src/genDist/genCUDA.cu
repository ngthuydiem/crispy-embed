 /***********************************************
* # Copyright 2011. Thuy Diem Nguyen
* # Contact: thuy1@e.ntu.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

#include "genMain.h"
#include "genKernel.h"
#include <thrust/host_vector.h>

void writeVectorToFile_GPU(thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector, thrust::host_vector< float > h_distVector, string outFileName, unsigned long long count, int fileId) {
	FILE * distFile; 
	
	distFile = fopen(outFileName.c_str(), "wb");
	if (distFile == NULL){
		printf("cannot open distFile: %s\n", outFileName.c_str());
		exit(-1);
	}

	float distArray[BUF_SIZE];	

	int h = 0;
	thrust::pair<unsigned int, unsigned int> aPair;						
	
				
	for (unsigned int i = 0; i < count; ++i)
	{					

		distArray[h] = h_distVector[i];	
		++h;		
	
		if (h == BUF_SIZE) {					
			fwrite(distArray, sizeof(float), BUF_SIZE, distFile);		
			h = 0;
		}	
	}
	
	if (h > 0) {					
		fwrite(distArray, sizeof(float), h, distFile);
		h = 0;
	}	
		
	fclose(distFile);				
}

void computeGenDist_CUDA(FILE* inFile, string outFileName, READ* &readArray, int numReads, int maxLen, float threshold, int band)
{			
	int i, maxDigitalLen = 0;
	int EOFTag = 0;
	int numPairs = 0;
	unsigned long long totalNumPairs = 0, count = 0;
	int fileId = 0;
	
	// check for the number of available GPUs	
	//printf("blockSize: %d, gridSize: %d\n", (int)BLOCK_SIZE, (int)GRID_SIZE);

	maxDigitalLen = (int) (maxLen / SUBMAT_DIM) + 2;
	if (maxDigitalLen % 16 != 0) 
		maxDigitalLen += 16 - (maxDigitalLen % 16);
	// padding
	//printf("maxDigitalLen: %d\n", maxDigitalLen);

	ushort *binaryReadArray;
	binaryReadArray = (ushort*) malloc(numReads * maxDigitalLen * sizeof(ushort));
	
	// Digitialize a sequence into bit storage format(on host, in an array of unsigned UINT).
	convertToBinary(readArray, numReads, binaryReadArray, maxDigitalLen);

	// clean up readArray
	for (i = 0; i < numReads; ++i)
		readArray[i].release();	
	free(readArray);
	
	// Allocate space for the pair id array	
	int *h_pairArray;
	int *d_pairArray;
	checkCudaErrors( cudaMallocHost((void**)&h_pairArray, NUM_PAIRS * 2 * sizeof(int)) );		
	checkCudaErrors( cudaMalloc((void**)&d_pairArray, NUM_PAIRS * 2 * sizeof(int)) );	

	// Allocate memory block for the Needleman-Wunsch distance array	
	float *h_distArray;	
	float *d_distArray;	
	checkCudaErrors( cudaMallocHost((void**)&h_distArray, NUM_PAIRS * sizeof(float)) );
	checkCudaErrors( cudaMalloc((void**)&d_distArray, NUM_PAIRS * sizeof(float)) );	


	// determine gridSize and blockSize
	size_t threadsPerBlock(BLOCK_SIZE);
	size_t blocksPerGrid(GRID_SIZE);

	int offset, chunkSize, numPairsPerChunk;
	
	// use cudaArray to store tupleArraySet
	cudaChannelFormatDesc channelDesc=cudaCreateChannelDesc<ushort>();
	cudaArray *seqCuArray;
	size_t width, height;
	width = maxDigitalLen*16;
	height = numReads/16;
	if ( (numReads&15) != 0) 
	 	++height;
	//cout << "2D texture: width " << width << " height: " << height << endl;
		 	
	checkCudaErrors( cudaMallocArray(&seqCuArray, &channelDesc, width, height) );	
	checkCudaErrors( cudaMemcpyToArray(seqCuArray, 0, 0, binaryReadArray, numReads * maxDigitalLen * sizeof(ushort), cudaMemcpyHostToDevice) );
	checkCudaErrors( cudaBindTextureToArray(getSeqTexRef(), seqCuArray, channelDesc) );

	free(binaryReadArray);

	cudaStream_t stream[NUM_STREAMS];
	for (i = 0; i < NUM_STREAMS; ++i)
		checkCudaErrors( cudaStreamCreate(&stream[i]) );
	
	size_t maxNumPairs = numReads*128;
	if (maxNumPairs > MAX_NUM_PAIRS)
		maxNumPairs = MAX_NUM_PAIRS;
	thrust::host_vector< float > h_distVector (maxNumPairs);	
	thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector (maxNumPairs * 2);	
	float dist;
	// obtain file size:
	
	while (!EOFTag)
	{
		numPairs = loadPairs(inFile, h_pairArray, EOFTag);	
		numPairsPerChunk = (numPairs + NUM_STREAMS - 1) / NUM_STREAMS;	
		
		if (numPairsPerChunk % 16 != 0) 
			numPairsPerChunk += 16 - (numPairsPerChunk % 16);
			

		for (i = 0; i < NUM_STREAMS; ++i) {
			
			offset = i * numPairsPerChunk;
			
			if (i < NUM_STREAMS - 1)
				chunkSize = numPairsPerChunk;
			else
				chunkSize = numPairs - offset;
					
			checkCudaErrors( cudaMemcpyAsync(d_pairArray+offset*2, h_pairArray+offset*2, chunkSize * sizeof(int) * 2, cudaMemcpyHostToDevice, stream[i]) );
		}
				
		for (i = 0; i < NUM_STREAMS; ++i) {
			
			offset = i * numPairsPerChunk;
			
			if (i < NUM_STREAMS - 1)
				chunkSize = numPairsPerChunk;
			else
				chunkSize = numPairs - offset;								
		
			if (band > 1) 
				launchGenKernel_band(stream[i], blocksPerGrid, threadsPerBlock, d_pairArray+offset*2, d_distArray+offset, maxDigitalLen, chunkSize, band);
			else 
				launchGenKernel_full(stream[i], blocksPerGrid, threadsPerBlock, d_pairArray+offset*2, d_distArray+offset, maxDigitalLen, chunkSize);				
		}
		
		for (i = 0; i < NUM_STREAMS; ++i) {
			
			offset = i * numPairsPerChunk;
			
			if (i < NUM_STREAMS - 1)
				chunkSize = numPairsPerChunk;
			else
				chunkSize = numPairs - offset;								
					
			// copy results from device to host
			checkCudaErrors( cudaMemcpyAsync(h_distArray+offset, d_distArray+offset, chunkSize * sizeof(float), cudaMemcpyDeviceToHost, stream[i]) );		
		}
		
		cudaDeviceSynchronize();
		
		for (i = 0; i < numPairs; i++) {
			dist = h_distArray[i];	
			if (dist < threshold || fabs(dist-threshold) < EPSILON)
			{
				h_pairVector[count] = thrust::make_pair(h_pairArray[i*2], h_pairArray[i*2+1]);
				h_distVector[count] = dist;

				++count;
			}	
		}
		
		if (count >= maxNumPairs)
		{			
			//h_pairVector.resize(count);
			//h_distVector.resize(count);	
	
			writeVectorToFile_GPU(h_pairVector, h_distVector, outFileName, count, fileId);	
			
			//h_pairVector.resize(maxNumPairs * 2);
			//h_distVector.resize(maxNumPairs * 2);	
	
			++ fileId;
			totalNumPairs += count;
			count = 0;										
		}			
	}
	
	
	if (count > 0)
	{
		h_pairVector.resize(count);
		h_distVector.resize(count);	
				
		writeVectorToFile_GPU(h_pairVector, h_distVector, outFileName, count, fileId);	
				
		totalNumPairs += count;										
	}	
	
	//printf("totalNumPairs: %llu\n", totalNumPairs);	
	printf("%llu\n", totalNumPairs);	
	
	for (i = 0; i < NUM_STREAMS; ++i)
		checkCudaErrors( cudaStreamDestroy(stream[i]) );

	checkCudaErrors( cudaFreeHost(h_distArray) );					
	checkCudaErrors( cudaFreeHost(h_pairArray) );
			
	checkCudaErrors( cudaFree(d_pairArray) );	
	checkCudaErrors( cudaFree(d_distArray) );				
	
	// clean up device variables
	checkCudaErrors( cudaUnbindTexture(getSeqTexRef()) );
	checkCudaErrors( cudaFreeArray(seqCuArray) );		
}


