#ifndef _EMBED_H_
#define _EMBED_H_

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <omp.h>
#include <sys/time.h>
using namespace std;

#define DEFAULT_NUM_READS 	1000000
#define BUF_SIZE	4096

// define READ

typedef struct read {
	int id;
	int length;
	char* sequence;
	int numTuples;
	unsigned int* tuples;

	void initialize(int readId, const char* seq, int K);
	void finalize();
	void formTuples(int K);
	void sortTuples();

} READ;


int compareTwoTuples(const void* t1, const void* t2);

void removeNewLine(string &line);

void removeInvalidChar(string &line);

int readFile(string inFileName, READ* &readArray, int &maxLen, int K);

void usage();

void getCommandOptions(int argc, char* argv[], string &inFileName, string &outFileName, int &K, int &groupSize, float &threshold, int &numThreads);

#endif
