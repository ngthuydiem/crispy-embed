#include "findSeeds.h"

float computeKDist(READ *readArray, int K, int id1, int id2) {

	unsigned short x, y, length1, length2, matches;
	float dist;
	int k,l;
	
	matches = 0;
	length1 = readArray[id1].length - K + 1;
	length2 = readArray[id2].length - K + 1;

	for (k = 0, l = 0; (k < length1) && (l < length2);)
	{
		x = readArray[id1].tuples[k];
		y = readArray[id2].tuples[l];

		matches = matches + (unsigned short)(x==y);
		k = k + (unsigned short)(x<=y);
		l = l + (unsigned short)(x>=y);	
	}

	dist = 1.0f - (float) matches / ( (length1 < length2) ? length1 : length2 );	
	return dist;
}

float computeTotalKDist(READ *readArray, int K, unordered_set<int> idSet, int id2) {

	float totalDist = 0, dist;
	unordered_set<int>::iterator it;
		
	for (it = idSet.begin(); it != idSet.end(); ++it)
	{
		dist = computeKDist(readArray, K, *it, id2);
		totalDist += dist;
	}
	return totalDist;
}

void printIdSet(unordered_set<int> idSet) {
	unordered_set<int>::iterator it;
	cout << "idSet size: " << idSet.size() << endl;
	for (it = idSet.begin(); it != idSet.end(); ++it)
		cout << *it << "\t";
	cout << endl;
}

int main(int argc, char* argv[]) {

	clock_t startTime, endTime;
	
	startTime = clock();
	READ *readArray = NULL;	

	string inFileName, outFileName;
	int numReads, numSeeds, maxLen;
	unordered_set<int> seeds;
	unordered_set<int> tempSeeds;
	unordered_set<int> finalSeeds;
	unordered_set<int>::iterator it1, it2, it;		
	int i, h, counter, interval, aSeed, newSeed;

	int pairArray[BUF_SIZE*2];	
	float dist, maxDist;
	int K = 5;
	
	getCommandOptions(argc, argv, inFileName, outFileName, K);	
		
	// read from input files
	numReads = readFile(inFileName, readArray, maxLen, K);			
	
	// open the output file
	FILE* outFile = NULL;			
	outFile = fopen(outFileName.c_str(), "w");	

	numSeeds = (int)log2(numReads);
	numSeeds = pow(numSeeds,2);	
	
	interval = numReads/numSeeds;
	counter = 0;

	for (i = 0; i < numReads; ++i) 		
		if (i % interval == 0 && counter < numSeeds)
		{
			seeds.insert(i);			
			++counter;
		}

	cout << "\tK = " << K << endl;							
	printIdSet(seeds);		
	
	for (i = 0; i < numReads; ++i) {
		readArray[i].formTuples(K);
		readArray[i].sortTuples();
	}
		
	finalSeeds = seeds;

	for (it = seeds.begin(); it != seeds.end(); ++it) 
	{
		aSeed = *it;

		maxDist = -1.0f;
		newSeed = 0;
		for (i = 0; i < numReads; ++i) 		
		{		
			dist = computeKDist(readArray, K, aSeed, i);
				
			if (dist > maxDist) {
				maxDist = dist;
				newSeed = i;
			}
		}

		aSeed = newSeed;
					
		maxDist = -1.0f;
		newSeed = 0;
		for (i = 0; i < numReads; ++i) 		
		{
			dist = computeKDist(readArray, K, aSeed, i);					
			
			if (dist > maxDist) {
				maxDist = dist;
				newSeed = i;
			}
		}
		finalSeeds.insert(newSeed);
	}
			
	tempSeeds = finalSeeds;	

	int id1, id2;		
	for (it1 = tempSeeds.begin(); it1 != tempSeeds.end(); ++it1) 
		for (it2 = it1; it2 != tempSeeds.end(); ++it2) 
	{		
			id1 = *it1;
			id2 = *it2;
			if (id1 != id2) 
			{			
				dist = computeKDist(readArray, K, id2, id1);
				if (dist <= 0.1)
				{
					if (readArray[id1].length < readArray[id2].length)
						finalSeeds.erase(id1);
					else
						finalSeeds.erase(id2);					
				}
			}
	}
	printIdSet(finalSeeds);		
	numSeeds = finalSeeds.size();
				
	h = 0;
	for (i = 0; i < numReads; ++i)
		for (it = finalSeeds.begin(); it != finalSeeds.end(); ++it) {
			
			pairArray[h*2] = i;
			pairArray[h*2+1] = *it;
			++h;
			if (h == BUF_SIZE) {
				fwrite(pairArray, sizeof(int),BUF_SIZE*2, outFile);
				h = 0;
			}
		}
	
	if (h > 0) 
		fwrite(pairArray, sizeof(int),h*2, outFile);

	for (int i = 0; i < numReads; i++)
		readArray[i].finalize();
	free(readArray);
	seeds.clear();
	tempSeeds.clear();
	finalSeeds.clear();
	fclose(outFile);
		
	endTime = clock();
	
	printf("\n----------------------------------------------------------------------\n");
	printf("                       	FIND SEEDS                           \n");
	printf("Name: %s. numReads: %d. numSeeds: %d\n", inFileName.c_str(), numReads, numSeeds);
	printf("BUF_SIZE = %d\n", BUF_SIZE);
	printf("Time taken: %.3f s\n", double(endTime - startTime) / CLOCKS_PER_SEC);
	printf("\n----------------------------------------------------------------------\n");
	
	return 0;
}

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
	this->tuples = (unsigned short*) calloc(this->numTuples, sizeof(unsigned short));

	// for each symbol in the sequence
	for (symbolIndex = 0, tupleIndex = 0; symbolIndex < this->length; symbolIndex++) {
		if (symbolIndex == 0)
			tuples[tupleIndex] = 0;
		if (symbolIndex >= K) {
			++tupleIndex;
			this->tuples[tupleIndex] = (this->tuples[tupleIndex - 1] << (2 * (9
				- K)));
			this->tuples[tupleIndex] = (this->tuples[tupleIndex] >> (2
				* (8 - K)));
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

	qsort(this->tuples, this->numTuples, sizeof(short int), compareTwoTuples);
}

void usage()
{
	cout << "-i input file name" << endl;	
	cout << "-o output file name" << endl;	
	cout << "-n number of sampled reads, default: 10000" << endl;
}

void getCommandOptions(int argc, char* argv[], string &inFileName, string &outFileName, int &K)
{
	// process input arguments
	for (int i = 0; i < argc; i++)
	{
		if (strcmp("-i", argv[i]) == 0)
			inFileName.assign(argv[i + 1]);		
		if (strcmp("-o", argv[i]) == 0)
			outFileName.assign(argv[i + 1]);	
		if (strcmp("-k", argv[i]) == 0)
			K = atoi(argv[i+1]);
		if (strcmp("-h", argv[i]) == 0) {
			usage();
			exit(0);
		}
	}
	// process input and output files
	if (inFileName.length() < 2 || outFileName.length() < 2)
	{
		usage();
		exit(0);
	}	
}

int compareTwoTuples(const void* t1, const void* t2) {

	return (*(short int*) t1 - *(short int*) t2);

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
