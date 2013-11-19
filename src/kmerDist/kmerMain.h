/***********************************************
* # Copyright 2011. Thuy Diem Nguyen
* # Contact: thuy1@e.ntu.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

#ifndef _KMER_MAIN_H_
#define _KMER_MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <string.h>

using namespace std;

#define BUF_SIZE		4096
#define BLOCK_SIZE		128
#define GRID_SIZE		BLOCK_SIZE * 64
#define NUM_STREAMS		4
#define NUM_PAIRS		BLOCK_SIZE * GRID_SIZE * NUM_STREAMS

#define EPSILON 			0.00001
#define DEFAULT_NUM_READS 	1000000
#define MAX_READ_LEN		4096
#define MAX_NUM_READS		2097152

#define max(a, b) ((a)>(b)?a:b)
#define min(a, b) ((a)<(b)?a:b)

// define READ

typedef struct read {
	int id;
	int length;
	char* sequence;
	int numTuples;
	unsigned short* tuples;

	void initialize(int readId, const char* seq, int K);
	void finalize();
	void formTuples(int K);
	void sortTuples();

} READ;

int loadPairs(FILE* pairFile, int * &pairArray, int &EOFTag);

// define function prototypes

int compareTwoTuples(const void* t1, const void* t2);

void removeNewLine(string &line);

void removeInvalidChar(string &line);

int readFile(string inFileName, READ* &readArray, int &maxLen, int K);

void writeToFile(FILE * distFile, float * h_distArray, int numPairs, long & totalNumPairs);

void help();

void getCommandOptions(int argc, char* argv[], string &inFileName, bool &useGPU, int &K);

void computeKmerDist_CPU(READ* &readArray, FILE* pairFile, FILE* distFile, int numReads, int K);

void computeKmerDist_CUDA(READ* &readArray, FILE* pairFile, FILE* distFile, int numReads, int maxLen, int K);

#endif
