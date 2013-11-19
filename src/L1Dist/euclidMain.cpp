/***********************************************
* # Copyright 2011. Nguyen Thuy Diem
* # Contact: Kristin
* #          thuy1@e.ntu.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

#include "euclidMain.h"

int main(int argc, char* argv[]) {

	struct timeval startTime, endTime;
	
	gettimeofday(&startTime, NULL);
	
	string inFileName, inDistName, pairFileName, distFileName;
	bool useGPU = true;
	float threshold = THRESHOLD;	

	unsigned long long totalNumPairs = 0;
	int i, numReads, numSeeds, arrayDim, readSize, numThreads=1;
	// open the output file
	getCommandOptions(argc, argv, inFileName, threshold, useGPU, numThreads, numReads);
	
	inDistName = inFileName;
	pairFileName = inFileName;
	distFileName = inFileName;
	inDistName.append(".dist");
	pairFileName.append(".epair");
	distFileName.append(".edist");
	
	FILE * inDistFile;
	inDistFile = fopen(inDistName.c_str(), "rb");
	if (inDistFile == 0)
		exit(-1);
													
	bool EOFTag = false;
	
	float distArray[BUF_SIZE];		
	while (!EOFTag)
	{			
		readSize = loadDistFile(inDistFile, distArray, EOFTag);					
		totalNumPairs += readSize;
	}
	rewind(inDistFile);
	numSeeds = totalNumPairs/numReads;

	float ** eReads;
	eReads = (float**) malloc(numReads * sizeof(float*));
	for (i = 0; i < numReads; ++i)
		eReads[i] = (float*) malloc(numSeeds * sizeof(float));
		
	// compute embedding vectors	
	for (i = 0; i < numReads; ++i) 
		fread(eReads[i], sizeof(float), numSeeds, inDistFile);
		
	arrayDim = (int)(16 * pow(8.0, floor(log10((double)numReads))-1));	 

	if (arrayDim > 1536)
		arrayDim = 1536;
		
	/*	
	printf("\n----------------------------------------------------------------------\n");
	printf("                 COMPUTE EUCLID DISTANCES                           \n");
	printf("File name: %s. numReads: %d. numSeeds: %d. threshold: %.2f\n\n", inFileName.c_str(), numReads, numSeeds, threshold);	
	*/				
	if (useGPU) {
		//printf("USE GPU. numStreams: %d\n", NUM_STREAMS);
		computeEuclidDist_CUDA(eReads, pairFileName, distFileName, numReads, numSeeds, threshold, arrayDim);					
	}
	else {	
		//printf("USE CPU. numThreads: %d\n", numThreads);
		
		omp_set_num_threads(numThreads);
		computeEuclidDist_CPU(eReads, pairFileName, distFileName, numReads, numSeeds, threshold);
	}

	fclose(inDistFile);			

	gettimeofday(&endTime, NULL);	
	long elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000u + (endTime.tv_usec - startTime.tv_usec) / 1.e3 + 0.5;
	/*
	printf("Time taken: %.3f s\n", elapsedTime/1.e3);
	printf("\n----------------------------------------------------------------------\n");
	*/
	return 0;
}

unsigned int loadDistFile(FILE* distFile, float* distArray, bool &EOFTag)
{
	size_t readSize;

	readSize = fread(distArray, sizeof(float), BUF_SIZE , distFile);	

	if (readSize < BUF_SIZE)
		EOFTag = true;

	return readSize;
}

void Trag_reverse_eq(int index, int N, int& row, int& col) 
{ 
   row = 0; 
   int keyafter; 
   do 
   { 
       row++; 
       keyafter = row * N - (row - 1) * row / 2; 
   } while (index >= keyafter); 
   row--; 
   col = N - keyafter + index; 
} 

void help()
{
	cout << "<-i inFileName>: FASTA file " << endl;	
	cout << "[-t threshold] : value range: 0.0 to 1.0, default: 0.5" << endl;
	cout << "[-n numThreads]: value range: 1 to maxNumProcs, default: 1" << endl;
	cout << "[-d]		    : output euclid distances " << endl;
	cout << "[-c]           : use CPU instead of GPU" << endl;
	cout << "[-m]           : use MPI version for GPU cluster" << endl;
	cout << "[-h]		    : help" << endl;
}

void getCommandOptions(int argc, char* argv[], string &inFileName, float &threshold, bool &useGPU, int &numThreads, int &numReads)
{		
	numThreads = omp_get_num_procs();
	numReads = 0;					
		
	// process input arguments
	for (int i = 0; i < argc; ++i)
	{
		if (strcmp("-i", argv[i]) == 0) 
		{
			inFileName.assign(argv[i + 1]);		
			// check input file
			if (inFileName.length() < 2)
			{
				help();
				exit(-1);				
			}		
		}
		if (strcmp("-n", argv[i]) == 0) {
			numReads = atoi(argv[i + 1]);				
			if (numReads < 1)
			{
				help();
				exit(-1);
			}
		}
		if (strcmp("-t", argv[i]) == 0) {
			threshold = (float)atof(argv[i + 1]);			
			// check distance threshold 
			if (threshold < 0.0f || threshold > 1.0f)
			{
				cout << "Warning: invalid distance threshold (-t option). Set to " << THRESHOLD << "."<< endl;
				threshold = THRESHOLD;
			}
		}
		if (strcmp("-c", argv[i]) == 0)
			useGPU = false;
		if (strcmp("-h", argv[i]) == 0) {
			help();
			exit(-1);
		}
	}	
}




