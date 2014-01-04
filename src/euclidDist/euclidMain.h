/***********************************************
* # Copyright 2011. Thuy Diem Nguyen
* # Contact: thuy1@e.ntu.edu.sg
* #
* # GPL 3.0 applies.
* #
* ************************************************/

#ifndef _EUCLID_MAIN_H_
#define _EUCLID_MAIN_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>

using namespace std;

#define BLOCK_DIM 				16	
#define NUM_STREAMS				4

#define THRESHOLD 				0.1
#define EPSILON 				0.00001
#define BUF_SIZE				4096
#define MAX_NUM_PAIRS 			1024*1024*32

void removeNewLine(string &line);

void removeInvalidChar(string &line);

void help();

int loadFreq(string freqFileName);

unsigned int loadDistFile(FILE* distFile, float* distArray, bool &EOFTag);

void Trag_reverse_eq(int index, int N, int& row, int& col);

void getCommandOptions(int argc, char* argv[], string &inFileName, float &threshold, bool &useGPU, int &numThreads, int &numReads);

void computeEuclidDist_CPU(float ** eReads, string pairFileName, string distFileName, int numReads, int numSeeds, float threshold);

void computeEuclidDist_CUDA(float ** eReads, string pairFileName, string distFileName, int numReads, int numSeeds, float threshold, int arrayDim);

#endif
