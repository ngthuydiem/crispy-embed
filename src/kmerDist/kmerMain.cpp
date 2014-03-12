/***********************************************
* # Copyright 2011. Nguyen Thuy Diem
* # Contact: Kristin
* #          thuy1@e.ntu.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

#include "kmerMain.h"

int main(int argc, char* argv[]) {

	READ *readArray = NULL;	

	clock_t startTime, endTime;
	
	startTime = clock();
	string inFileName, pairFileName, distFileName;
	bool useGPU = true;

	int numReads, maxLen = 0, K = 4;
	// open the output file
	FILE* pairFile = NULL, *distFile = NULL;
		
	getCommandOptions(argc, argv, inFileName, useGPU, K);

	pairFileName = inFileName;
	pairFileName.append(".pair");
	distFileName = inFileName;
	distFileName.append(".dist");
	
	// read from input files
	numReads = readFile(inFileName, readArray, maxLen, K);		
	
	if (numReads > MAX_NUM_READS || maxLen > MAX_READ_LEN) {
		printf("Error: unsupported numReads or maxLen: %d, %d. Exit...", numReads, maxLen);
		exit(-1);
	}

	// form the k-tuple array for each sequence from the file
	for (int i = 0; i < numReads; ++i) {
		readArray[i].formTuples(K);
		readArray[i].sortTuples();
	}
			
	printf("\n----------------------------------------------------------------------\n");
	printf("                      COMPUTE KMER DISTANCES                           \n");
	printf("File name: %s\n\n", inFileName.c_str());
	printf("numReads: %d, maxLen: %d, K: %d\n", numReads, maxLen, K);	

	pairFile = fopen(pairFileName.c_str(), "rb");
	distFile = fopen(distFileName.c_str(), "wb");	

	if (useGPU) {
		printf("USE GPU\n");
		computeKmerDist_CUDA(readArray, pairFile, distFile, numReads, maxLen, K);					
	}
	else {	
		printf("USE CPU\n");		
		computeKmerDist_CPU(readArray, pairFile, distFile, numReads, K);
	}
			
	fclose(pairFile);
	fclose(distFile);
	
	endTime = clock();
	printf("Time taken: %.3f s\n", double(endTime - startTime) / CLOCKS_PER_SEC);
	printf("\n----------------------------------------------------------------------\n");
			
	return 0;
}

int loadPairs(FILE* pairFile, int * &pairArray, int &EOFTag)
{
	size_t readSize;

	readSize = fread(pairArray, sizeof(int), NUM_PAIRS*2, pairFile);	

	if (readSize < NUM_PAIRS*2)
		EOFTag = 1;

	return readSize/2;
}

// implementation of READ

void READ::initialize(int readId, const char* seq, int K) {

	this->id = readId;
	this->length = strlen(seq);
	this->numTuples = this->length - K + 1;
	if (this->length < 2)
		cout << "Error: empty sequence!\n" << endl;
	this->sequence = (char *) malloc(sizeof(char) * (strlen(seq) + 1));
	strcpy(this->sequence, seq);	
}

void READ::finalize() {

	if (this->tuples!=NULL)
		free(this->tuples);
	this->tuples = NULL;
}

// digital forms of 4 symbols: A:00, G:01, T:10, C:11

void READ::formTuples(int K) {

	// calculate the number of tuples for each sequence
	int symbolIndex, tupleIndex;
	this->tuples = (unsigned int*) calloc(this->numTuples, sizeof(unsigned int));

	// for each symbol in the sequence
	for (symbolIndex = 0, tupleIndex = 0; symbolIndex < this->length; symbolIndex++) {
		if (symbolIndex == 0)
			tuples[tupleIndex] = 0;
		if (symbolIndex >= K) {
			++tupleIndex;
			this->tuples[tupleIndex] = (this->tuples[tupleIndex - 1] << (2 * (17
				- K)));
			this->tuples[tupleIndex] = (this->tuples[tupleIndex] >> (2
				* (16 - K)));
		} else {
			this->tuples[tupleIndex] = (this->tuples[tupleIndex] << 2);
		}

		switch (this->sequence[symbolIndex]) {
		case 'A': // 00
			break;
		case 'G': // 01
			this->tuples[tupleIndex] |= 1;
			break;
		case 'T': // 10
			this->tuples[tupleIndex] |= (1 << 1);
			break;
		case 'C': // 11
			this->tuples[tupleIndex] |= (1 << 1);
			this->tuples[tupleIndex] |= 1;
			break;
		default: 
			break;
		}
	}
}

void READ::sortTuples() {

	qsort(this->tuples, this->numTuples, sizeof(unsigned int), compareTwoTuples);
}

void writeToFile(FILE * distFile, float * h_distArray, int numPairs, long & totalNumPairs)
{
	float distArray[BUF_SIZE];
	int h = 0;
	
	// write result to output file
	for (int i = 0; i < numPairs; ++i) 
	{
		if (h_distArray[i] > 0 || fabs(h_distArray[i]) < EPSILON)
		{						
			if (h == BUF_SIZE) {					
				fwrite(distArray, sizeof(float), BUF_SIZE, distFile);		
				totalNumPairs += BUF_SIZE;
				h = 0;
			}
			distArray[h] = h_distArray[i];
			++h;						
		}
	}
	if (h > 0) {					
		fwrite(distArray, sizeof(float), h, distFile);	
		totalNumPairs += h;
	}
}


void help()
{
	cout << "<-i inFileName>: FASTA file " << endl;	
	cout << "[-n numThreads]: value range: 1 to maxNumProcs, default: 1" << endl;
	cout << "[-c]           : use CPU instead of GPU" << endl;
	cout << "[-h]		    : help" << endl;
}

void getCommandOptions(int argc, char* argv[], string &inFileName, bool &useGPU, int &K)
{		
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
		if (strcmp("-c", argv[i]) == 0)
			useGPU = false;
		if (strcmp("-k", argv[i]) == 0)
			K = atoi(argv[i+1]);
		if (strcmp("-h", argv[i]) == 0) {
			help();
			exit(-1);
		}
	}	
}


int compareTwoTuples(const void* t1, const void* t2) {

	return (*(unsigned int*) t1 - *(unsigned int*) t2);

}

//--------------------- Trim the buff, de-whitespaces --------------------
void removeNewLine(string &line) {
	size_t found;
	found = line.find_last_not_of(" \n\r\t\v\f\b");
	if (found != string::npos)
		line = line.substr(0, found + 1);
}

void removeInvalidChar(string &line) {
	size_t found;
	found = line.find_last_of("ATGCatgc");
	if (found != string::npos)
		line = line.substr(0, found + 1);
}


//--------------------------------------------------------------------------
int readFile(string inFileName, READ* &readArray, int &maxLen, int K) {

	FILE * inFile;
	char buffer[BUF_SIZE];

	inFile = fopen(inFileName.c_str(), "rb");
	if (inFile == NULL) {
		cout << "Error: cannot open read file: " << inFileName << " Exit..." << endl;
		exit(-1);
	}

	readArray = (READ *) malloc(DEFAULT_NUM_READS * sizeof(READ));

	if (readArray == NULL) {
		cout << "Error: not enough memory. Require extra: " << DEFAULT_NUM_READS * sizeof(READ) / (1024 * 1024) << "MB. Exit..." << endl;		
		fclose(inFile);
		exit(-1);
	}

	string tempStr;
	int readIndex = 0;
	string seq = "", label;
	int stopFlag = 0;

	// initialise the reading process
	fgets(buffer, BUF_SIZE, inFile);
	tempStr.clear();
	tempStr.assign(buffer);
	if (tempStr[0] != '>') {
		cout << "Error: read file is not in FASTA Format. Exit..."<< endl;
		fflush(stdin);
		exit(-1);
	} else {
		removeNewLine(tempStr);
		label.clear();
		label = tempStr;
	}

	// read from input file
	while (!feof(inFile)) {
		buffer[0] = '$';
		fgets(buffer, BUF_SIZE, inFile);
		tempStr.clear();
		tempStr.assign(buffer);
		removeNewLine(tempStr);
		if (tempStr[0] != '>' && tempStr[0] != '$') {
			removeInvalidChar(tempStr);
			seq += tempStr;
			continue;
		} else if (seq.length() == 0) {
			if (buffer[0] == '$') {
				stopFlag = 1;
				break;
			}
			continue;
		} else {
			removeInvalidChar(seq);
			// if there are more sequences than the buffer length
			if (readIndex + 1 > DEFAULT_NUM_READS) 
				readArray = (READ *) realloc(readArray, (readIndex + 1) * sizeof(READ));
			if (readArray == NULL) {
				cout << "Error: not enough memory. Exit..."<< endl;
				fclose(inFile);
				exit(-1);
			}

			if (seq.length() > maxLen)
				maxLen = seq.length();
				
			readArray[readIndex].initialize(readIndex, seq.c_str(), K);
			readIndex++;
			seq.clear();

			if (buffer[0] == '$') {
				stopFlag = 1;
				break;
			}
			removeNewLine(tempStr);
			label.clear();
			label = tempStr;
			continue;
		}
	}

	if (stopFlag == 0 && seq.length() != 0) {
		if (readIndex + 1 > DEFAULT_NUM_READS)
			readArray = (READ *) realloc(readArray, (readIndex + 1) * sizeof(READ));
		if (readArray == NULL) {
			cout << "Error: not enough memory. Exit..."<< endl;
			fclose(inFile);
			exit(-1);
		}
		removeInvalidChar(seq);
		readArray[readIndex].initialize(readIndex, seq.c_str(), K);
		readIndex++;
	}
	
	fclose(inFile);
	return readIndex;
}

