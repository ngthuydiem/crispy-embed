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
	
	cout << "write to : " << tempStr << " " << count << " pairs" << endl; 
				
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

void writeToVector(thrust::host_vector< thrust::pair<unsigned int, unsigned int> > & h_pairVector, thrust::host_vector< float > & h_distVector, float *h_distArray, int stageX, int stageY, int arrayDim, float threshold, unsigned long long & count) {

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
		dist = h_distArray[i];	
		if (dist < threshold || fabs(dist-threshold) < EPSILON)
		{
			h_pairVector[count] = thrust::make_pair(row, col);
			h_distVector[count] = dist;

			++count;
		}							
	}									
}

void computeEuclidDist_CUDA(float ** eReads, string pairFileName, string distFileName, int numReads, int numSeeds, float threshold, int arrayDim) {
	
	int i, j, row, index;	
	unsigned long long totalNumPairs = 0, count = 0;
	int fileId = 0;
	float dist;
	
	if (numReads % BLOCK_DIM != 0)
		numReads -= (numReads % BLOCK_DIM);
	
	// determine GRID_DIM and blockSize
	dim3 threadsPerBlock(BLOCK_DIM,BLOCK_DIM);	
	dim3 blocksPerGrid(numReads/BLOCK_DIM, numReads/BLOCK_DIM);	
		
	// declare host variables
	float *d_inDistSet;
	float *h_inDistSet;	

	checkCudaErrors( cudaMalloc((void**)&d_inDistSet, numReads * numSeeds * sizeof(float)) );	
	checkCudaErrors( cudaMallocHost((void**)&h_inDistSet, numReads * numSeeds * sizeof(float)) );	
	
	for (i = 0; i < numReads; ++i)
	{
		row = i * numSeeds;
		for (j = 0; j < numSeeds; ++j)
			h_inDistSet[row + j] = eReads[i][j];	
	}
		
	for (i = 0; i < numReads; ++i)
		free(eReads[i]);
	free(eReads);			

	checkCudaErrors( cudaMemcpyAsync(d_inDistSet, h_inDistSet,  numReads * numSeeds * sizeof(float), cudaMemcpyHostToDevice) );							
					
	// declare device variables
	float *d_distArray;
	float *h_distArray;	

	checkCudaErrors( cudaMalloc((void**)&d_distArray, numReads * numReads * sizeof(float)) );	
	checkCudaErrors( cudaMallocHost((void**)&h_distArray, numReads * numReads * sizeof(float)) );	
		
	thrust::host_vector< float > h_distVector (MAX_NUM_PAIRS_GPU * 2);	
	thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector (MAX_NUM_PAIRS_GPU * 2);									
						        								
	launchEuclidKernel(blocksPerGrid, threadsPerBlock, d_inDistSet, d_distArray, numReads, numSeeds);					
	checkCudaErrors( cudaMemcpyAsync(h_distArray, d_distArray, numReads * numReads * sizeof(float), cudaMemcpyDeviceToHost) );											
	cudaDeviceSynchronize();	
	// write result to output file
	for (i = 0; i < numReads; ++i) 
	{
		for (j = i+1; j < numReads; ++j) 
		{
			index = i * numReads + j;	
			dist = h_distArray[index];	

			if (dist < threshold || fabs(dist-threshold) < EPSILON)
			{
				h_pairVector[count] = thrust::make_pair(i, j);
				h_distVector[count] = dist;				
				++count;				
			}	
			//printf("%d %d %f %f\n", i, j, h_distArray[i * numReads + j], h_distArray[j * numReads + i]);						
		}
	}	
	
	h_pairVector.resize(count);
	h_distVector.resize(count);	
	
	writeVectorToFile_GPU(h_pairVector, h_distVector, pairFileName, distFileName, count, fileId);				
	totalNumPairs += count;																											
	
	// clean up host variables	
	checkCudaErrors( cudaFreeHost(h_distArray) );
	checkCudaErrors( cudaFree(d_distArray) );		
	
	checkCudaErrors( cudaFreeHost(h_inDistSet) );
	checkCudaErrors( cudaFree(d_inDistSet) );		
	
	printf("totalNumPairs: %llu\n", totalNumPairs);	
}

