/*
Copyright 2010 Marijn Kentie
 
This file is part of GASW.

GASW is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

GASW is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GASW.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "stdafx.h"
#include "main_cu.h"
#include "FastaFile.h"
#include "FastaMatrix.h"
#include "GPUdb.h"
#include "CPU.h"
#include "main.h"
#include "CommandLine.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <wchar.h>
#include <string.h>
#include <algorithm>

//#ifdef UNICODE 
//
//#define _tcslen     wcslen
//#define _tcscpy     wcscpy
//#define _tcscpy_s   wcscpy_s
//#define _tcsncpy    wcsncpy
//#define _tcsncpy_s  wcsncpy_s
//#define _tcscat     wcscat
//#define _tcscat_s   wcscat_s
//#define _tcsupr     wcsupr
//#define _tcsupr_s   wcsupr_s
//#define _tcslwr     wcslwr
//#define _tcslwr_s   wcslwr_s
//
//#define _stprintf_s swprintf_s
//#define _stprintf   swprintf
//#define _tprintf    wprintf
//
//#define _vstprintf_s    vswprintf_s
//#define _vstprintf      vswprintf
//
//#define _tscanf     wscanf
//
//
//#define TCHAR wchar_t
//
//#else
//
//#define _tcslen     strlen
//#define _tcscpy     strcpy
//#define _tcscpy_s   strcpy_s
//#define _tcsncpy    strncpy
//#define _tcsncpy_s  strncpy_s
//#define _tcscat     strcat
//#define _tcscat_s   strcat_s
//#define _tcsupr     strupr
//#define _tcsupr_s   strupr_s
//#define _tcslwr     strlwr
//#define _tcslwr_s   strlwr_s
//
//#define _stprintf_s sprintf_s
//#define _stprintf   sprintf
//#define _tprintf    printf
//
//#define _vstprintf_s    vsprintf_s
//#define _vstprintf      vsprintf
//
//#define _tscanf     scanf
//
//#define TCHAR char
//#endif
//
Options options;

struct Result
{
	int index;
	int score;
};

/**
Check CUDA device capabilities
*/
static bool checkDevice()
{
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,0);
		
	if(props.warpSize/2!=GPUdb::BLOCK_SIZE)
	{
		puts("Device half-warp size does not match database block size");
		//return false;
	}

	return true;
}

static bool resultComparisonFunc(Result r1, Result r2)
{
	return (r1.score>r2.score);
}

int main(int argc, char* argv[])
{

	options.gapWeight = -10;
	options.extendWeight = -2;
	options.listSize = 20;
	options.dna = 0;
	
	//printf("\nSTART OF THE PROGRAM\n");

	if(!parseCommandLine(argc, argv))
		return EXIT_FAILURE;

	//Show help info if not enough command line options provided
	if(options.sequenceFile == 0 || options.dbFile == 0)
	{
		puts("Usage: gpu [options] <sequence> <database>");
		puts("Sequence: FASTA format file; Database: DBCONV .gpudb file.");
		puts("Options:");
		printArgs();
		return EXIT_SUCCESS;
	}

	//Show settings that will be used
	printf("Sequence: %s\nDatabase: %s\n",options.sequenceFile,options.dbFile);
printf("1\n");

	printArgValues();
	printf("2\n");

	puts("");

	if(!checkDevice())
	{
		puts("Error: device does not meet requirements.");
		return false;
	}

	clock_t tInit = clock();
	//Load query
	FastaFile inFile;
	if(!inFile.load(options.sequenceFile,options.dna))
	{
		puts("Error reading sequence file.");
		return EXIT_FAILURE;
	}
	//printf("\nDEBUG\n");
	printf("%s: input sequence length is %d.\n",options.sequenceFile,inFile.getSequenceLength(0));

	//Load substitution matrix
	FastaMatrix matrix;
	if(!matrix.load(options.matrix))

	{
		puts("Error loading substitution matrix");
		return EXIT_FAILURE;
	}
	matrix.createQueryProfile(inFile,options.dna);
	
	//Load database
	GPUdb dbFile;
	puts("Loading database...");
	if(!dbFile.load(options.dbFile))
	{
		puts("Error reading database file.");
		return EXIT_FAILURE;
	}
	char descsFile[FILENAME_MAX];
	strcpy(descsFile,options.dbFile);
	strcat(descsFile,".descs");
	if(!dbFile.loadDescriptions(descsFile))
	{
		puts("Error reading database descriptions file.");
		return EXIT_FAILURE;
	}
	printf("%s: %d symbols in %d sequence(s) in %d block(s) in database.\n",options.dbFile,dbFile.getNumSymbols(),dbFile.getNumSequences(),dbFile.getNumBlocks());

	clock_t stop = clock();
	double totalInit=(stop-tInit)/(double)CLOCKS_PER_SEC;
	printf("Init: %f\n",totalInit);

	//Launch kernel
	scoreType* scores=0;
	size_t queryLength = inFile.getSequenceLength(0); 
	double numCells = queryLength*(double)dbFile.getNumSymbols();
	puts("\nLaunching kernel.");
	fflush(stdout);
	
	double seconds;
	bool success= launchSW(&scores, inFile.getSequence(0),queryLength,dbFile,matrix, options.gapWeight, options.extendWeight, seconds);
	
	if(!success)
	{
		cudaError_t err =cudaGetLastError();
		printf("Error in Smith-Waterman function: %d\n",err);
		return EXIT_FAILURE;
	}

	printf("Done. Seconds: %f, GCUPS: %f\n\n",seconds, numCells/seconds/(1000*1000*1000));

	//Check if kernel set all scores
	for(size_t i=0;i<dbFile.getNumSequences();i++)
	{
		if(scores[i]==(scoreType)-1)
			puts("SCORE NOT SET!!!");
	}

	//Sort scores
	puts("Sorting results...");	
	std::vector<Result> sortScores;
	sortScores.resize(dbFile.getNumSequences());
	for(size_t i=0;i<sortScores.size();i++)
	{
		sortScores[i].index = i;
		sortScores[i].score = scores[i];
	}
	cudaFreeHost(scores);
	std::sort(sortScores.begin(),sortScores.end(),&resultComparisonFunc);
	
	//Display results
	puts("Results:");
	for(size_t i=0;i < std::min((int)options.listSize,(int)dbFile.getNumSequences());i++)
	{
		printf("%3d. %-50.50s\t SCORE: %d\n",i,dbFile.getDescription(sortScores[i].index),sortScores[i].score);
	}

	//Write top scoring sequences to library file so they can be aligned by SSearch
	if(options.topSequenceFile)
	{
		dbFile.openOutFile(options.topSequenceFile);
		for(size_t i=0;i<options.listSize&&i<sortScores.size();i++)
		{
			dbFile.writeToFile(sortScores[i].index);
		}
		dbFile.closeOutFile();
	}

	#ifdef _DEBUG
	getchar();
	#endif
	return EXIT_SUCCESS;
}


