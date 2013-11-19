#include "findSeeds.h"

float computeKDist(READ *readArray, int kValue, int id1, int id2) {

	unsigned short x, y, length1, length2, matches;
	float dist;
	int k,l;
	
	matches = 0;
	length1 = readArray[id1].length - kValue + 1;
	length2 = readArray[id2].length - kValue + 1;

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

float computeTotalKDist(READ *readArray, int kValue, unordered_set<int> idSet, int id2) {

	float totalDist, dist;
	unsigned short x, y, length1, length2, matches;
	int i, k,l, id1;
		
	unordered_set<int>::iterator it;
	int count = 0;
	
	int *id_array;
	id_array = (int*) calloc(idSet.size(),sizeof(int));
	for (it = idSet.begin(); it != idSet.end(); ++it)
	{
		id_array[count] = *it;
		count ++;
	}
	
	totalDist = 0.0;
#pragma omp parallel for private(id1, k, l, length1, length2, matches, x, y, dist) reduction(+:totalDist) schedule(static,1)
	for (i = 0; i < count; i++)
	{
		id1 = id_array[i];
		matches = 0;
		length1 = readArray[id1].length - kValue + 1;
		length2 = readArray[id2].length - kValue + 1;

		for (k = 0, l = 0; (k < length1) && (l < length2);)
		{
			x = readArray[id1].tuples[k];
			y = readArray[id2].tuples[l];

			matches = matches + (unsigned short)(x==y);
			k = k + (unsigned short)(x<=y);
			l = l + (unsigned short)(x>=y);	
		}

		dist = 1.0f - (float) matches / ( (length1 < length2) ? length1 : length2 );	
		totalDist += dist;
	}

	free(id_array);
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

	struct timeval startTime, endTime;
	
	gettimeofday(&startTime, NULL);
	READ *readArray = NULL;	

	string inFileName, outFileName;
	int numReads, numSeeds, maxLen;
	unordered_set<int> seeds;
	unordered_set<int> tempSeeds;
	unordered_set<int> finalSeeds;
	unordered_set<int>::iterator it1, it2, it;		
	int i, h, counter, interval, aSeed, newSeed;

	int pairArray[BUF_SIZE*2];
	bool flag;
	float dist, maxDist, threshold;
	int kValue, groupSize, numThreads;
	
	getCommandOptions(argc, argv, inFileName, outFileName, kValue, groupSize, threshold, numThreads);	
		
	omp_set_num_threads(numThreads);
	// read from input files
	numReads = readFile(inFileName, readArray, maxLen, kValue);			
	
	// open the output file
	FILE* outFile = NULL;			
	outFile = fopen(outFileName.c_str(), "w");	

	float t = log2(numReads);		
	numSeeds = (int) (t*t);	
	numSeeds = min(numSeeds, numReads);		
	
	interval = numReads/numSeeds;
	counter = 0;

	for (i = 0; i < numReads; ++i) 		
		if (i % interval == 0 && counter < numSeeds)
		{
			seeds.insert(i);			
			++counter;
		}

#pragma omp parallel for schedule(static,1)	
	for (i = 0; i < numReads; ++i) {
		readArray[i].formTuples(kValue);
		readArray[i].sortTuples();
	}
				
	for (it = seeds.begin(); it != seeds.end(); ++it) 
	{
		flag = true;
		tempSeeds.clear();
		aSeed = *it;
		tempSeeds.insert(aSeed);
			
		do
		{
			maxDist = -1.0f;
			newSeed = 0;
			for (i = 0; i < numReads; ++i) 		
			{		
				dist = computeTotalKDist(readArray, kValue, tempSeeds, i);
				
				if (dist > maxDist) {
					maxDist = dist;
					newSeed = i;
				}
			}
				
			if (find(tempSeeds.begin(), tempSeeds.end(), newSeed) == tempSeeds.end())
			{
				tempSeeds.insert(newSeed);
				flag = true;
			}
			else
				flag = false;
		} while(flag && tempSeeds.size() <= groupSize);	
			
		finalSeeds.insert(tempSeeds.begin(), tempSeeds.end());
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
				dist = computeKDist(readArray, kValue, id2, id1);
				if (dist <= threshold)
				{
					if (readArray[id1].length < readArray[id2].length)
						finalSeeds.erase(id1);
					else
						finalSeeds.erase(id2);							
				}
			}
	}
	
	//printIdSet(finalSeeds);		
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
		
	gettimeofday(&endTime, NULL);	
	long elapsedTime = (endTime.tv_sec - startTime.tv_sec) * 1000u + (endTime.tv_usec - startTime.tv_usec) / 1.e3 + 0.5;
	
	printf("\n----------------------------------------------------------------------\n");
	printf("                       	FIND SEEDS                           \n");
	printf("Name: %s. numReads: %d. numSeeds: %d. numThreads: %d\n", inFileName.c_str(), numReads, numSeeds, numThreads);
	cout << "groupSize = " << groupSize << "\tK = " << kValue << endl;			
	printf("Time taken: %.3f s\n", elapsedTime/1.e3);
	printf("\n----------------------------------------------------------------------\n");
	
	return 0;
}

void READ::initialize(int readId, const char* seq, int kValue) {

	this->id = readId;
	this->length = strlen(seq);
	this->numTuples = this->length - kValue + 1;
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

void READ::formTuples(int kValue) {

	// calculate the number of tuples for each sequence
	int symbolIndex, tupleIndex;
	this->tuples = (unsigned short*) calloc(this->numTuples, sizeof(unsigned short));
	for (symbolIndex = 0, tupleIndex = 0; symbolIndex < this->length; symbolIndex++) {
		if (symbolIndex == 0)
			tuples[tupleIndex] = 0;
		if (symbolIndex >= kValue) {
			++tupleIndex;
			this->tuples[tupleIndex] = (this->tuples[tupleIndex - 1] << (2 * (9
				- kValue)));
			this->tuples[tupleIndex] = (this->tuples[tupleIndex] >> (2
				* (8 - kValue)));
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
}

void getCommandOptions(int argc, char* argv[], string &inFileName, string &outFileName, int &kValue, int &groupSize, float &threshold, int &numThreads)
{
	int numProcs = omp_get_num_procs();					
	if (numProcs > 2)
		numThreads = numProcs/2;
	else
		numThreads = 1;
		
	threshold = 0.03;
	kValue = 6;
	groupSize = 2;
		
	// process input arguments
	for (int i = 0; i < argc; i++)
	{
		if (strcmp("-i", argv[i]) == 0)
			inFileName.assign(argv[i + 1]);		
		if (strcmp("-o", argv[i]) == 0)
			outFileName.assign(argv[i + 1]);	
		if (strcmp("-k", argv[i]) == 0)
			kValue = atoi(argv[i+1]);
		if (strcmp("-t", argv[i]) == 0)
			threshold = atof(argv[i+1]);			
		if (strcmp("-s", argv[i]) == 0)
			groupSize = atoi(argv[i+1]);
		if (strcmp("-n", argv[i]) == 0) {
			numThreads = atoi(argv[i + 1]);				
			if (numThreads < 1 || numThreads > numProcs)
			{
				cout << "Warning: invalid number of threads (-n option). Set to " << numThreads << "." << endl;
			}
		}	
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


int readFile(string inFileName, READ* &readArray, int &maxLen, int kValue) {

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
				
			readArray[readIndex].initialize(readIndex, seq.c_str(), kValue);
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
		readArray[readIndex].initialize(readIndex, seq.c_str(), kValue);
		readIndex++;
	}
	
	fclose(inFile);
	return readIndex;
}
