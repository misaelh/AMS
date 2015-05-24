/**
Database converter
*/

//#include <tchar.h>
#include <cmath>
#include "../gpu/GPUdbConverter.h"
#include "../gpu/GPUdb.h"
#include <string.h>
int main(int argc, char* argv[])
{
	bool dna = false;

	if(argc<2)
	{
		puts("Converts FASTA database to gpudb file.");
		puts("Usage: dbconv [options] <database>");
		puts("Options: -d for DNA instead of protein databases.");
		puts("Database: FASTA format file.");		
		return 0;
	}

	if(pow(2.0f,(float)GPUdb::LOG2_BLOCK_SIZE)!=GPUdb::BLOCK_SIZE)
	{
		puts("Error: program compiled with 2^LOG2_BLOCK_SIZE!=BLOCK_SIZE");
		return 1;
	}

	if(argc>2)
	{
		if(strcmp(argv[1],"-d")==0)
		{
			dna=true;
			puts("DNA MODE");
		}
		else
		{
			puts("Unknown command line option.");
			return 1;
		}
	}

	printf("Conversion parameters: block size %d, sub block size %d, alignment %d.\n",GPUdb::BLOCK_SIZE,GPUdb::SUBBLOCK_SIZE,GPUdb::ALIGNMENT);
	
	puts("Loading...");
	char* dbFile;
	if(dna)
		dbFile = argv[2];
	else
		dbFile = argv[1];
	GPUdbConverter db;	
	if(!db.load(dbFile,dna))
	{
		puts("Error opening file.");
		return 1;
	}
	wprintf(L"%ls: %d symbols in %d sequence(s) in database.\n",dbFile,db.getNumSymbols(),db.getNumSequences());

	//db.trim();
	puts("Converting...");
	db.convert();

	char *outFile = "out.gpudb";
	wprintf(L"Writing %lws\n",outFile);
	if(!db.write(outFile))
	{
		puts("Error writing output file.");
		return 1;
	}

	char *descsFile = "out.gpudb.descs";
	wprintf(L"Writing %ls\n",descsFile);
	if(!db.writeDescriptions(descsFile))
	{
		puts("Error writing description file.");
		return 1;
	}
	/*
	GPUdb gdb;
	gdb.load(L"out.gpudb");
	gdb.openOutFile(L"out.lseg");
	for(size_t i=0;i<gdb.getNumSequences();i++)
		gdb.writeToFile(i);
	gdb.closeOutFile();
	*/

	puts("Done.");
	//getchar();
	return 0;
}
