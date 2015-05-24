#pragma once
#include "FastaFile.h"
#include "GPUdb.h"
#include <vector>

class GPUdbConverter : public FastaFile
{
private:
	static FastaRecord padding;
	struct Block
	{
		size_t length;
		std::vector<FastaRecord*> sequenceGroups[GPUdb::BLOCK_SIZE];
	};
	std::vector<Block> blocks;	
	size_t* sequenceOffsets;
	size_t* sequenceNumbers;
	size_t totalNumSequences;
	char* blob;
	size_t blobSize;

	struct 
	{
		size_t numPadding;
		size_t numFillSequences;
	} stats;


public:	
	GPUdbConverter();
	~GPUdbConverter();
	bool convert();
	void arrangeSequences();
	void createSequenceBlob();
	bool write(const char* fileName);
	void writeBlock(char* ptr, size_t block);
	void writeSequenceGroup(char* ptr, std::vector<FastaRecord*> &group, size_t groupNum);
	bool writeDescriptions(const char* fileName);
};
