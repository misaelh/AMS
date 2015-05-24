#pragma once
#include "FastaFile.h"
#include <fstream>
#include <cstddef>

//Round x up so it's a whole amount of y
#define WHOLE_AMOUNT_OF(x,y) ((x+y-1)/y)
#define PADDING_SEQ_NAME "PADDING | PADDING | PADDING"

class GPUdb
{
public:
	static const size_t BLOCK_SIZE = 16; /**< How many sequences make up a block. Should be device half-warp size. Should be power of 2. */
	static const size_t LOG2_BLOCK_SIZE = 4; /**< To be able to use a shift instead of division in the device code */
	static const size_t ALIGNMENT = 256;
	static const size_t SUBBLOCK_SIZE = 8; /**< The number of symbols of each sequence to interleave at a time; to be able to read multiple symbols in one access */
	static const char SEQUENCE_GROUP_TERMINATOR = ' ';
	static const char SUB_SEQUENCE_TERMINATOR = '#';
	typedef unsigned int blockOffsetType;
	typedef unsigned int seqSizeType;
	typedef unsigned int seqNumType;
	typedef char seqType;

	struct GroupData /**< Per-sequence group data */
	{
		seqSizeType seqSize;
		seqSizeType seqNum;
	};

private:
	char* blob; /**< Whole database loaded into memory */
	char* d_blob; /**< Device-side database */
	size_t blobSize; /**< Size in bytes */
	
	/** Pointers to host and device side sequence lenghts and sequence letters in the blob */
	blockOffsetType* blockOffsets;
	seqNumType* seqNums;
	seqType* sequences;
	seqNumType* d_seqNums;
	blockOffsetType* d_blockOffsets;
	seqType* d_sequences;

	char* descriptionBuffer; /**< All descriptions */
	std::vector<char*> descriptions; /**< Pointers into description buffer */
	struct
	{
		size_t numSequences;
		size_t numSymbols;
		size_t numBlocks;
		size_t alignmentPadding1;
		size_t alignmentPadding2;		
	} metadata; /**< General info on the database */

	size_t* sequenceOffsets; /**< Pointers to sequences in blob */

	std::ofstream outFile;

public:

	GPUdb();
	~GPUdb();
	bool load(const char* fileName);
	bool loadDescriptions(const char* fileName);
	size_t getNumSequences() const;
	size_t getDBSizeInBytes() const;
	size_t getNumSymbols() const;
	size_t getNumBlocks() const;
	size_t getSequenceLength(size_t sequenceNum) const;
	const char* getDescription(unsigned int index) const;
	
	bool copyToGPU();
	const blockOffsetType *get_d_BlockOffsets();
	const seqNumType *get_d_SeqNums();
	const seqType *get_d_Sequences();

	bool openOutFile(const char *fileName);
	bool writeToFile(size_t sequenceNum);
	void closeOutFile();

};
