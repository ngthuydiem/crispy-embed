/***********************************************
* # Copyright 2011. Thuy Diem Nguyen & Zejun Zheng
* # Contact: thuy1@e.ntu.edu.sg or zheng_zejun@sics.a-star.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

// Note: don't use_fast_math option

#include "euclidMain.h"
#include "euclidKernel.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>	

void writeVectorToFile_GPU(thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector, thrust::host_vector< float > h_distVector, string pairFileName, string distFileName, unsigned long long count, int fileId) {
	FILE * pairFile, * distFile; 
	string tempStr;	
	char buf[1000];
					
	sprintf(buf, "_%d", fileId);

	tempStr = pairFileName;
	tempStr.append(buf);
	pairFile = fopen(tempStr.c_str(), "wb");
	if (pairFile == NULL){
		printf("cannot open pairFile: %s\n", tempStr.c_str());
		exit(-1);
	}	
	tempStr = distFileName;
	tempStr.append(buf);
	distFile = fopen(tempStr.c_str(), "wb");
	if (distFile == NULL){
		printf("cannot open distFile: %s\n", tempStr.c_str());
		exit(-1);
	}
				
	thrust::device_vector<float> d_distVector = h_distVector; 
	thrust::device_vector< thrust::pair<unsigned int, unsigned int> > d_pairVector = h_pairVector;
	
	thrust::sort_by_key(d_distVector.begin(), d_distVector.end(), d_pairVector.begin());
				
	thrust::copy(d_distVector.begin(), d_distVector.end(), h_distVector.begin());
	thrust::copy(d_pairVector.begin(), d_pairVector.end(), h_pairVector.begin());
								
	int pairArray[BUF_SIZE*2];
	float distArray[BUF_SIZE];	

	int h = 0;
	thrust::pair<unsigned int, unsigned int> aPair;						
	
	//cout << "write to : " << tempStr << " " << count << " pairs" << endl; 
				
	for (unsigned int i = 0; i < count; ++i)
	{					
		aPair = h_pairVector[i];	
		distArray[h] = h_distVector[i];
		pairArray[h*2] = aPair.first;
		pairArray[h*2+1] = aPair.second;		
		++h;		
		if (h == BUF_SIZE) {					
			fwrite(pairArray, sizeof(unsigned int), BUF_SIZE * 2, pairFile);		
			fwrite(distArray, sizeof(float), BUF_SIZE, distFile);		
			h = 0;
		}	
	}
	
	if (h > 0) {					
		fwrite(pairArray, sizeof(unsigned int), h * 2, pairFile);		
		fwrite(distArray, sizeof(float), h, distFile);
		h = 0;
	}	
		
	fclose(pairFile);
	fclose(distFile);				
}

void writeToVector(thrust::host_vector< thrust::pair<unsigned int, unsigned int> > & h_pairVector, thrust::host_vector< float > & h_distVector, float *h_out, int stageX, int stageY, int arrayDim, float threshold, unsigned long long & count) {

	int i, row, col, rowOffset, colOffset;	
	float dist;		
	int arraySize = arrayDim * arrayDim;
	
	rowOffset = stageX * arrayDim;
	colOffset = stageY * arrayDim;				

	// write result to output file
	for (i = 0; i < arraySize; ++i) 
	{
		row = rowOffset + (int)i / arrayDim;
		col = colOffset + (int)i % arrayDim;	
		dist = h_out[i];	
		//if (dist < threshold || fabs(dist-threshold) < EPSILON)
		if (dist < threshold)
		{
			h_pairVector[count] = thrust::make_pair(row, col);
			h_distVector[count] = dist;

			++count;
		}							
	}									
}

void computeEuclidDist_CUDA(float ** eReads, string pairFileName, string distFileName, int numReads, int numSeeds, float threshold, int arrayDim) {
	
	int i, j, stageX, stageY, row, offset, stageId;	
	unsigned long long totalNumPairs = 0, count = 0;
	int fileId = 0;
	
	int size = arrayDim * arrayDim;	
	int arraySize = size * NUM_STREAMS;
	int gridSize = (arrayDim + BLOCK_DIM - 1)/BLOCK_DIM;	
	int stageDim =  (numReads + arrayDim - 1)/arrayDim;
	
	// determine GRID_DIM and blockSize
	dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM);	
	dim3 blocksPerGrid(gridSize, gridSize);	
/*		
	// get number of SMs on this GPU
	printf("size: %dx%d, arraySize: %d, stageDim: %dx%d\n", arrayDim, arrayDim, arraySize, stageDim, stageDim);	
	printf("blockSize: %dx%d, gridSize: %dx%d\n", BLOCK_DIM, BLOCK_DIM, gridSize, gridSize); 	
*/
	// declare host variables
	float *h_in;
	float *d_in;
	checkCudaErrors( cudaMalloc((void**)&d_in, numReads * numSeeds * sizeof(float)) );	
	checkCudaErrors( cudaMallocHost((void**)&h_in, numReads * numSeeds * sizeof(float)) );
	
	for (i = 0; i < numReads; ++i)
	{
		row = i * numSeeds;
		for (j = 0; j < numSeeds; ++j)
			h_in[row + j] = eReads[i][j];	
	}
/*
	float *mean, *variance, diff;
	mean = (float*) malloc(numSeeds * sizeof(float));
	variance = (float*) malloc(numSeeds * sizeof(float));
	
	for (j = 0; j < numSeeds; ++j) {
		mean[j] = 0.0f;
		variance[j] = 0.0f;
	}
		
	for (i = 0; i < numReads; ++i) 
		for (j = 0; j < numSeeds; ++j) 
			mean[j] += eReads[i][j];
	
	for (j = 0; j < numSeeds; ++j) 
		mean[j] /= numSeeds;
		
	for (i = 0; i < numReads; ++i) 
		for (j = 0; j < numSeeds; ++j) 
		{
			diff = eReads[i][j]-mean[j];
			variance[j] += diff * diff;			
		}
	
	for (j = 0; j < numSeeds; ++j) 
		variance[j] /= numReads;
	
	float *d_var;
	checkCudaErrors( cudaMalloc((void**)&d_var, numSeeds * sizeof(float)) );
	checkCudaErrors( cudaMemcpyAsync(d_var, variance, numSeeds * sizeof(float), cudaMemcpyHostToDevice) );				
	free(mean);
	free(variance);	
*/	
	for (i = 0; i < numReads; ++i)
		free(eReads[i]);
	free(eReads);	
	
	checkCudaErrors( cudaMemcpyAsync(d_in, h_in, numReads * numSeeds * sizeof(float), cudaMemcpyHostToDevice) );									

	// declare device variables
	float *d_out;
	float *h_out;	

	checkCudaErrors( cudaMalloc((void**)&d_out, arraySize * sizeof(float)) );	
	checkCudaErrors( cudaMallocHost((void**)&h_out, arraySize * sizeof(float)) );
				
	cudaStream_t streams[NUM_STREAMS];

	for (i = 0; i < NUM_STREAMS; ++i) 
		checkCudaErrors( cudaStreamCreate(&streams[i]) );			

	thrust::host_vector< float > h_distVector (MAX_NUM_PAIRS_GPU * 2);	
	thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector (MAX_NUM_PAIRS_GPU * 2);

	int stageSize = stageDim * (stageDim + 1) / 2;												
	
	
	for (j = 0; j < stageSize; j += NUM_STREAMS)
	{		

		for (i = 0; i < NUM_STREAMS; ++i) {
			offset = i * size;		
			stageId = i + j;
			
			if (stageId < stageSize) {
				Trag_reverse_eq(stageId, stageDim, stageX, stageY);													
						        								
				//launchEuclidKernel(streams[i], blocksPerGrid, threadsPerBlock, d_in, d_out+offset, numReads, numSeeds, stageX, stageY, arrayDim, d_var);	
				launchEuclidKernel(streams[i], blocksPerGrid, threadsPerBlock, d_in, d_out+offset, numReads, numSeeds, stageX, stageY, arrayDim);	
				
				checkCudaErrors( cudaMemcpyAsync(h_out+offset, d_out+offset, size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]) );							
				 		
			}
		}		

		cudaDeviceSynchronize();
		
		for (i = 0; i < NUM_STREAMS; ++i) {								
			offset = i * size;		
			stageId = i + j;				

			if (stageId < stageSize) {							
							
				Trag_reverse_eq(stageId, stageDim, stageX, stageY);		
				
				writeToVector(h_pairVector, h_distVector, h_out+offset, stageX, stageY, arrayDim, threshold, count);
																											
			}
		}				
		
		if (count >= MAX_NUM_PAIRS_GPU)
		{			
			h_pairVector.resize(count);
			h_distVector.resize(count);	
	
			writeVectorToFile_GPU(h_pairVector, h_distVector, pairFileName, distFileName, count, fileId);	
			
			h_pairVector.resize(MAX_NUM_PAIRS_GPU * 2);
			h_distVector.resize(MAX_NUM_PAIRS_GPU * 2);	

			++ fileId;
			totalNumPairs += count;
			count = 0;										
		}				
	}	
	
	if (count > 0)
	{
			h_pairVector.resize(count);
			h_distVector.resize(count);	
				
			writeVectorToFile_GPU(h_pairVector, h_distVector, pairFileName, distFileName, count, fileId);	
				
			totalNumPairs += count;										
	}	
	
	
	for (i = 0; i < NUM_STREAMS; ++i) 
		checkCudaErrors( cudaStreamDestroy(streams[i]) );

	//checkCudaErrors( cudaFree(d_var) );
					
	// clean up host variables	
	checkCudaErrors( cudaFreeHost(h_out) );
	checkCudaErrors( cudaFree(d_out) );		
	
	checkCudaErrors( cudaFreeHost(h_in) );
	checkCudaErrors( cudaFree(d_in) );		
	
	//printf("totalNumPairs: %llu\n", totalNumPairs);	
	printf("%llu\n", totalNumPairs);	
}


