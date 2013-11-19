/***********************************************
* # Copyright 2011. Thuy Diem Nguyen
* # Contact: thuy1@e.ntu.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

#include "genMain.h"
#include <thrust/host_vector.h>

void writeVectorToFile_CPU(thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector, thrust::host_vector< float > h_distVector, string outFileName, unsigned long long count, int fileId) {
	FILE * distFile; 
	
	distFile = fopen(outFileName.c_str(), "wb");
	if (distFile == NULL){
		printf("cannot open distFile: %s\n", outFileName.c_str());
		exit(-1);
	}			
								
	float distArray[BUF_SIZE];	

	int h = 0;				
				
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

bool insideBand(int i, int j, int kBand)
{
	return ((i - j >= -kBand) && (i - j <= kBand));
}

// TODO: wrong
void computeGenDist_CPU_full(FILE* inFile, string outFileName, READ* &readArray, int numReads, int maxLen, float threshold, int numThreads)
{
	int pairIndex=0, i, j, numPairs;	
	unsigned long long totalNumPairs = 0, count = 0;
	int fileId=0;

	int **M, **ML, **AL;
	char **ULD;	
	
	// allocate memory
	M = (int**) Malloc(sizeof(int*) * numThreads * 2);
	ML = (int**) Malloc(sizeof(int*) * numThreads * 2);
	AL = (int**) Malloc(sizeof(int*) * numThreads * 2);
	ULD = (char**) Malloc(sizeof(char*) * numThreads * 2);

	for (i = 0; i < numThreads * 2; ++i) {
		M[i] = (int*) Malloc(sizeof(int) * (maxLen+1));
		ML[i] = (int*) Malloc(sizeof(int) * (maxLen+1));
		AL[i] = (int*) Malloc(sizeof(int) * (maxLen+1));
		ULD[i] = (char*) Malloc(sizeof(char) * (maxLen+1));	
	}

	int M_up, M_left, M_diag, alignedScore;
	int length1, length2, readIndex1, readIndex2;	

	bool isMatched;
	float dist;
	int EOFTag = 0;
	int n;
	int curRow, preRow;
	int curRowFlag, preRowFlag;
	int maxMLastRow, maxMLastCol, MLRow, MLCol, ALRow, ALCol;

	// Allocate space for the pair id array	
	int *kmerPairArray;
	kmerPairArray = (int*)malloc(NUM_PAIRS * 2 * sizeof(int));
	if (kmerPairArray == NULL) {
		cout << "Error: not enough memory! EXIT..." << endl;
		fclose(inFile);
		exit(-1);
	}		

	thrust::host_vector< float > h_distVector (MAX_NUM_PAIRS_CPU * 2);	
	thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector (MAX_NUM_PAIRS_CPU * 2);	

	while (EOFTag != 1)
	{
		numPairs = loadPairs(inFile, kmerPairArray, EOFTag);				

		#pragma omp parallel for private(n, i, j, length1, length2, readIndex1, readIndex2, isMatched, alignedScore, M_up, M_left, curRow, preRow, curRowFlag, preRowFlag, dist, maxMLastRow, maxMLastCol, MLRow, MLCol, ALRow, ALCol) schedule(static,1)
		for (pairIndex = 0; pairIndex < numPairs; ++pairIndex)
		{	
			maxMLastRow = maxMLastCol = 0;		
			MLRow = MLCol = ALRow = ALCol = 0;
			n = omp_get_thread_num();
			curRowFlag = 0, preRowFlag = 1;
			curRow = curRowFlag + 2 * n;
			preRow = preRowFlag + 2 * n;
			readIndex1 = kmerPairArray[pairIndex * 2];
			readIndex2 = kmerPairArray[pairIndex * 2 + 1];

			length1 = readArray[readIndex1].length;
			length2 = readArray[readIndex2].length;

			M[curRow][0] = 0;
			ULD[curRow][0] = 0;
			ML[curRow][0] = 0;
			AL[curRow][0] = 0;

			// Initialise the first row
			for (j = 1; j <= length1; ++j)
			{
				M[curRow][j] = 0;
				ULD[curRow][j] = 0;
				ML[curRow][j] = 0;
				AL[curRow][j] = j;			
			}
			
			// Recurrence relations
			// Process row by row
			for (i = 1; i <= length2; ++i)
			{
				curRowFlag = 1 - curRowFlag;
				preRowFlag = 1 - preRowFlag;
				curRow = curRowFlag + 2 * n;
				preRow = preRowFlag + 2 * n;

				M[curRow][0] = 0;
				ULD[curRow][0] = 0;
				ML[curRow][0] = 0;
				AL[curRow][0] = i;

				for (j = 1; j <= length1; ++j)
				{
					if (readArray[readIndex2].sequence[i - 1] == readArray[readIndex1].sequence[j - 1])
					{
						alignedScore = MATCH;
						isMatched = 1;
					}
					else
					{
						alignedScore = MISMATCH;
						isMatched = 0;
					}

					M_up = M[curRow][j-1] + (ULD[curRow][j-1] & 1) * GO + (ULD[curRow][j-1] >> 2 & 1) * GE;
					M_left = M[preRow][j] + (ULD[preRow][j] & 1) * GO + (ULD[preRow][j] >> 1 & 1) * GE;
					M_diag = M[preRow][j-1] + alignedScore;
					M[curRow][j] = max(max(M_diag, M_up), M_left);

					if (M[curRow][j] == M_diag) {
						ULD[curRow][j] = 1;
						ML[curRow][j] = (ML[preRow][j-1] - isMatched) + 1;
						AL[curRow][j] = AL[preRow][j-1] + 1;						
					}
					else if (M[curRow][j] == M_up) {
						ULD[curRow][j] = 4;
						ML[curRow][j] = ML[curRow][j-1] + 1;
						AL[curRow][j] = AL[curRow][j-1] + 1;
					}
					else {
						ULD[curRow][j] = 2;
						ML[curRow][j] = ML[preRow][j]  + 1;
						AL[curRow][j] = AL[preRow][j] + 1;
					}						

					if (i == length2 && M[curRow][j] >= maxMLastCol) {
						maxMLastCol = M[curRow][j];		
						MLCol = ML[curRow][j];
						ALCol = AL[curRow][j];
					}	

					if (j == length1 && M[curRow][j] >= maxMLastRow) {
						maxMLastRow = M[curRow][j];
						MLRow = ML[curRow][j];
						ALRow = AL[curRow][j];					
					}													
				}				
			}			
			
			if (maxMLastCol > maxMLastRow)
				dist = (float)MLCol/ALCol;
			else 
				dist = (float)MLRow/ALRow;

			#pragma omp critical 
			if (dist < threshold || fabs(dist-threshold) < EPSILON)
			{
				h_pairVector[count] = thrust::make_pair(i,j);
				h_distVector[count] = dist;
				++count;
			}	
		}
		
		if (count >= MAX_NUM_PAIRS_CPU)
		{		
			h_pairVector.resize(count);
			h_distVector.resize(count);			
			
			writeVectorToFile_CPU(h_pairVector, h_distVector, outFileName, count, fileId);	
			
			++ fileId;
			totalNumPairs += count;
			count = 0;										
			
			h_pairVector.resize(MAX_NUM_PAIRS_CPU * 2);
			h_distVector.resize(MAX_NUM_PAIRS_CPU * 2);	
		}		
	}
	
	if (count > 0)
	{			
		h_pairVector.resize(count);
		h_distVector.resize(count);			

		writeVectorToFile_CPU(h_pairVector, h_distVector, outFileName, count, fileId);	
		totalNumPairs += count;
	}		
	
	printf("totalNumPairs: %llu\n", totalNumPairs);

	// clean up
	for (i = 0; i < numReads; ++i)
		readArray[i].release();
	free(readArray);
	for (i = 0; i < numThreads * 2; ++i) {
		free(M[i]);
		free(AL[i]);
		free(ML[i]);
		free(ULD[i]);		
	}
	free(M);
	free(AL);
	free(ML);
	free(ULD);
	free(kmerPairArray);
}


void computeGenDist_CPU_band(FILE* inFile, string outFileName, READ* &readArray, int numReads, int maxLen, float threshold, int band, int numThreads)
{
	int pairIndex, i, j, numPairs;	
	unsigned long long totalNumPairs = 0, count = 0;
	int fileId = 0;

	int maxMLastRow, maxMLastCol, MLRow, MLCol, ALRow, ALCol;
	int M_up, M_left, M_diag, alignedScore;	
	int length1, length2, readIndex1, readIndex2;	

	bool isMatched;
	float dist;
	int n, kBand;
	int curRow, preRow;
	int EOFTag = 0, curRowFlag, preRowFlag;

	int **M, **ML, **AL;
	char **ULD;	
	
	// allocate memory
	M = (int**) Malloc(sizeof(int*) * numThreads * 2);
	ML = (int**) Malloc(sizeof(int*) * numThreads * 2);
	AL = (int**) Malloc(sizeof(int*) * numThreads * 2);
	ULD = (char**) Malloc(sizeof(char*) * numThreads * 2);

	for (i = 0; i < numThreads * 2; ++i) {
		M[i] = (int*) Malloc(sizeof(int) * (maxLen+1));
		ML[i] = (int*) Malloc(sizeof(int) * (maxLen+1));
		AL[i] = (int*) Malloc(sizeof(int) * (maxLen+1));
		ULD[i] = (char*) Malloc(sizeof(char) * (maxLen+1));	
	}
	
	// Allocate space for the pair id array	
	int *kmerPairArray;
	kmerPairArray = (int*)malloc(NUM_PAIRS * 2 * sizeof(int));	
	if (kmerPairArray == NULL) {
		cout << "Memory error! EXIT..." << endl;
		fclose(inFile);
		exit(-2);
	}	

	thrust::host_vector< float > h_distVector (MAX_NUM_PAIRS_CPU * 2);	
	thrust::host_vector< thrust::pair<unsigned int, unsigned int> > h_pairVector (MAX_NUM_PAIRS_CPU * 2);	

	while (EOFTag != 1)
	{
		numPairs = loadPairs(inFile, kmerPairArray, EOFTag);		

		#pragma omp parallel for private(n, i, j, kBand, length1, length2, readIndex1, readIndex2, isMatched, alignedScore, M_up, M_left, curRow, preRow, curRowFlag, preRowFlag, dist, maxMLastRow, maxMLastCol, MLRow, MLCol, ALRow, ALCol) schedule(static,1)
		for (pairIndex = 0; pairIndex < numPairs; ++pairIndex)
		{
			maxMLastRow = maxMLastCol = 0;
			MLRow = MLCol = ALRow = ALCol = 0;
			n = omp_get_thread_num();
			curRowFlag = 0, preRowFlag = 1;
			curRow = curRowFlag + 2 * n;
			preRow = preRowFlag + 2 * n;
			readIndex1 = kmerPairArray[pairIndex * 2];
			readIndex2 = kmerPairArray[pairIndex * 2 + 1];

			length1 = readArray[readIndex1].length;
			length2 = readArray[readIndex2].length;
			// input is sorted by sequence length
			// hence length1 is always smaller than length2
			//kBand = min (length1, length2) / band;
			kBand =  (length1 + length2) / (band * 2);
			
			M[curRow][0] = 0;
			ULD[curRow][0] = 0;
			ML[curRow][0] = 0;
			AL[curRow][0] = 0;

			// Initialise the first row
			for (j = 1; j <= kBand; ++j)
			{
				M[curRow][j] = 0;
				ULD[curRow][j] = 0;
				ML[curRow][j] = 0;
				AL[curRow][j] = j;
			}

			// Recurrence relations
			// Process row by row
			for (i = 1; i <= length2; ++i)
			{
				curRowFlag = 1 - curRowFlag;
				preRowFlag = 1 - preRowFlag;
				curRow = curRowFlag + 2 * n;
				preRow = preRowFlag + 2 * n;

				if (i <= kBand)
				{
					M[curRow][0] = 0;
					ULD[curRow][0] = 0;
					ML[curRow][0] = 0;
					AL[curRow][0] = i;
				}

				for (j = 1; j <= length1; ++j)
				{
					if ( insideBand(i, j, kBand) )	
					{				
						if (readArray[readIndex2].sequence[i-1] == readArray[readIndex1].sequence[j-1])
						{
							alignedScore = MATCH;
							isMatched = 1;
						}
						else
						{
							alignedScore = MISMATCH;
							isMatched = 0;
						}
					
						if (insideBand(i,j-1,kBand))
							M_up = M[curRow][j-1] + (ULD[curRow][j-1] & 1) * GO + (ULD[curRow][j-1] >> 2 & 1) * GE;							
						else
							M_up = -1000000;
	
						if (insideBand(i-1,j,kBand))				
							M_left = M[preRow][j] + (ULD[preRow][j] & 1) * GO + (ULD[preRow][j] >> 1 & 1) * GE;
						else
							M_left = -1000000;
						
						M_diag = M[preRow][j-1] + alignedScore;
						M[curRow][j] = max(max(M_diag, M_up), M_left);

						if (M[curRow][j] == M_diag) {
							ULD[curRow][j] = 1;
							ML[curRow][j] = (ML[preRow][j-1] - isMatched) + 1;
							AL[curRow][j] = AL[preRow][j-1] + 1;						
						}
						else if (M[curRow][j] == M_up) {
							ULD[curRow][j] = 4;
							ML[curRow][j] = ML[curRow][j-1] + 1;
							AL[curRow][j] = AL[curRow][j-1] + 1;
						}
						else {
							ULD[curRow][j] = 2;
							ML[curRow][j] = ML[preRow][j]  + 1;
							AL[curRow][j] = AL[preRow][j] + 1;
						}		
	
						if (i == length2 && M[curRow][j] >= maxMLastCol) 							
						{							
							maxMLastCol = M[curRow][j];			
							MLCol = ML[curRow][j];
							ALCol = AL[curRow][j];
						}						
						
						if (j == length1 && M[curRow][j] >= maxMLastRow) 							
						{
							maxMLastRow = M[curRow][j];
							MLRow = ML[curRow][j];
							ALRow = AL[curRow][j];							
						}						
					}	
					else 
						M[curRow][j] = 0;
				}				
			}

			if (maxMLastCol > maxMLastRow)
				dist = (float)MLCol/ALCol;
			else
				dist = (float)MLRow/ALRow;	
							
			#pragma omp critical			
			if (dist < threshold || fabs(dist-threshold) < EPSILON)
			{
				h_pairVector[count] = thrust::make_pair(i,j);
				h_distVector[count] = dist;
				++count;
			}	
		}
		if (count >= MAX_NUM_PAIRS_CPU)
		{		
			h_pairVector.resize(count);
			h_distVector.resize(count);			
			
			writeVectorToFile_CPU(h_pairVector, h_distVector, outFileName, count, fileId);	
			
			++ fileId;
			totalNumPairs += count;
			count = 0;										
			
			h_pairVector.resize(MAX_NUM_PAIRS_CPU * 2);
			h_distVector.resize(MAX_NUM_PAIRS_CPU * 2);	
		}				
	}
	
	if (count > 0)
	{			
		h_pairVector.resize(count);
		h_distVector.resize(count);			

		writeVectorToFile_CPU(h_pairVector, h_distVector, outFileName, count, fileId);	
		totalNumPairs += count;
	}		

	printf("totalNumPairs: %lld\n", totalNumPairs);

	// clean up
	for (i = 0; i < numReads; ++i)
		readArray[i].release();
	free(readArray);
	for (i = 0; i < numThreads * 2; ++i) {
		free(M[i]);
		free(AL[i]);
		free(ML[i]);
		free(ULD[i]);		
	}
	free(M);
	free(AL);
	free(ML);
	free(ULD);
	free(kmerPairArray);
}


