#pragma once

struct Options
{
	char* sequenceFile;
	char* dbFile;
	char* matrix;
	char* topSequenceFile;
	int gapWeight;
	int extendWeight;
	unsigned int listSize;
	bool dna;
}; /**< Program options, globally accessible */

extern struct Options options;

