/***********************************************
* # Copyright 2011. Thuy Diem Nguyen
* # Contact: thuy1@e.ntu.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

#include "euclidMain.h"

#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>

void writeVectorToFile_CPU(thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector, thrust::host_vector< float > h_distVector, string pairFileName, string distFileName, unsigned long long count, int fileId) {
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
			
				
	thrust::sort_by_key(h_distVector.begin(), h_distVector.end(), h_pairVector.begin());				
								
	int pairArray[BUF_SIZE*2];
	float distArray[BUF_SIZE];	

	int h = 0;
	thrust::pair<unsigned int, unsigned int> aPair;						
				
	for (unsigned int i = 0; i < count; ++i)
	{					
		aPair = h_pairVector[i];	
		distArray[h] = h_distVector[i];
		pairArray[h*2] = aPair.first;
		pairArray[h*2+1] = aPair.second;		
		++h;		
	/*
		if (i <= 100)
			cout << aPair.first << "\t" << aPair.second << "\t" << distArray[i] << endl;	
	*/
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

void computeEuclidDist_CPU(float ** eReads, string pairFileName, string distFileName, int numReads, int numSeeds, float threshold, int arrayDim)
{
	int i, j, k;
	float diff, dist, sum;
	unsigned long long totalNumPairs = 0, count = 0;
	int fileId = 0;
	
	size_t maxNumPairs = arrayDim * arrayDim;
	thrust::host_vector< float > h_distVector (maxNumPairs * 2);	
	thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector (maxNumPairs * 2);	

	for (i = 0; i < numReads; ++i)
	{
		#pragma omp parallel 
		{	
		#pragma omp for nowait private(j, k, diff, sum, dist) schedule(static,1)	
		for (j = i+1; j < numReads; ++j)
		{		
			sum = 0;
			for (k = 0; k < numSeeds; ++k) {
				diff = eReads[i][k]-eReads[j][k];
				sum += diff * diff;
				
			}
			dist = sqrt(sum/numSeeds);

			#pragma omp critical			
			//if (dist < threshold || fabs(dist-threshold) < EPSILON)
			if (dist < threshold)
			{
				h_pairVector[count] = thrust::make_pair(i,j);
				h_distVector[count] = dist;
				++count;
			}			
		}	
		}
		
		if (count >= maxNumPairs)
		{		
			h_pairVector.resize(count);
			h_distVector.resize(count);			
			
			writeVectorToFile_CPU(h_pairVector, h_distVector, pairFileName, distFileName, count, fileId);	
			
			++ fileId;
			totalNumPairs += count;
			count = 0;										
			
			h_pairVector.resize(maxNumPairs * 2);
			h_distVector.resize(maxNumPairs * 2);	
		}		
	}
	if (count > 0)
	{			
		h_pairVector.resize(count);
		h_distVector.resize(count);			

		writeVectorToFile_CPU(h_pairVector, h_distVector, pairFileName, distFileName, count, fileId);	
		totalNumPairs += count;
	}		
						

	for (i = 0; i < numReads; i++)
		free(eReads[i]);
	free(eReads);	

	printf("%llu\n", totalNumPairs);		
}

