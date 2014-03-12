/***********************************************
* # Copyright 2011. Thuy Diem Nguyen
* # Contact: thuy1@e.ntu.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

#include "kmerMain.h"

void computeKmerDist_CPU(READ* &readArray, FILE* pairFile, FILE* distFile, int numReads, int K)
{
	int i, j, k, l;
	int numPairs, pairIndex, EOFTag;
	float dist;
	long totalNumPairs = 0;
	unsigned int x, y, length1, length2, matches;

	int *pairArray;
	pairArray = (int*) malloc(NUM_PAIRS * 2 * sizeof(int));

	float distArray[BUF_SIZE];

	int h = 0;
	
	EOFTag = 0;
	while (EOFTag != 1)
	{
		numPairs = loadPairs(pairFile, pairArray, EOFTag);			
		
		for (pairIndex = 0; pairIndex < numPairs; ++pairIndex)
		{		
			i = pairArray[pairIndex * 2];
			j = pairArray[pairIndex * 2 + 1];
			matches = 0;
			length1 = readArray[i].length - K + 1;
			length2 = readArray[j].length - K + 1;

			for (k = 0, l = 0; (k < length1) && (l < length2);)
			{
				x = readArray[i].tuples[k];
				y = readArray[j].tuples[l];

				matches = matches + (unsigned int)(x==y);
				k = k + (unsigned int)(x<=y);
				l = l + (unsigned int)(x>=y);	
			}

			dist = 1.0f - (float) matches / min(length1,length2);
			if (h == BUF_SIZE) {
				fwrite(distArray, sizeof(float),BUF_SIZE, distFile);
				totalNumPairs += BUF_SIZE;
				h = 0;
			}
			distArray[h] = dist;
			++h;				
		}
	}	

	if (h > 0) {
		fwrite(distArray, sizeof(float),h, distFile);
		totalNumPairs += h;					
	}
	
	for (i = 0; i < numReads; ++i)
		readArray[i].finalize();
	free(readArray);
	free(pairArray);	

	printf("totalNumPairs: %ld\n", totalNumPairs);	
}

