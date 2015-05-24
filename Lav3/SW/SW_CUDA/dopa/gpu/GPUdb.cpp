/**
Loader for binary GPU databases.

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
#include "GPUdb.h"
#include <limits>
#include <string.h>


GPUdb::GPUdb(): blob(0), blobSize(0), descriptionBuffer(0), d_blob(0), sequenceOffsets(0)
{
	metadata.numSequences = 0;
	metadata.numSymbols = 0;
}

GPUdb::~GPUdb()
{
	cudaFreeHost(blob);
	cudaFree(d_blob);
	delete [] descriptionBuffer;
	delete [] sequenceOffsets;
}

bool GPUdb::load(const char *fileName)
{
	std::ifstream file;
	file.open(fileName,std::ios::binary);
	if(!file.is_open())
		return false;	

	//Get metadata
	file.read((char*)&metadata,sizeof(metadata));	

	size_t offsetsSize = metadata.numSequences*sizeof(size_t);
	size_t headerSize = sizeof(metadata)+offsetsSize;

	//Get sequence offsets
	sequenceOffsets = new size_t[metadata.numSequences];
	file.read((char*)sequenceOffsets,offsetsSize);

	file.seekg(0, std::ios::end);
	blobSize = (size_t)file.tellg()-headerSize;
	file.seekg(0, std::ios::beg);

	//Read file into memory, skipping the metadata to guarantee alignment
	file.seekg(headerSize);
	//blob = (char*)malloc(blobSize);
	if(cudaHostAlloc(&blob,blobSize,cudaHostAllocWriteCombined)!=cudaSuccess)
		return false;
	file.read(blob,blobSize);
	file.close();

	blockOffsets = (blockOffsetType*) blob;
	seqNums = (seqNumType*) ((char*) blockOffsets + metadata.numBlocks*sizeof(blockOffsetType));
	seqNums = (seqNumType*) ((char*) seqNums + metadata.alignmentPadding1);
	sequences = (seqType*) ((char*) seqNums + metadata.numBlocks*GPUdb::BLOCK_SIZE*sizeof(seqNumType));
	sequences = (seqType*) ((char*) sequences + metadata.alignmentPadding2);

	return true;
}

bool GPUdb::loadDescriptions(const char *fileName)
{
	static const int DESC_SIZE = 100;
	
	std::ifstream file;
	file.open(fileName);
	if(!file.is_open())
		return false;
	descriptions.resize(metadata.numSequences);	
	descriptionBuffer = new char[metadata.numSequences*DESC_SIZE];
	char* ptr = descriptionBuffer;
	for(size_t i=0;i<metadata.numSequences;i++)
	{
		if(file.eof())
			return false;
		file.getline(ptr,DESC_SIZE);
		if(file.fail()) //Buffer full
		{
			file.clear();
			file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
		}
		descriptions[i]=ptr;
		ptr+=DESC_SIZE;
	}
	return true;

}

size_t GPUdb::getNumSequences() const
{
	return metadata.numSequences;
}

size_t GPUdb::getDBSizeInBytes() const
{
	return blobSize;
}

size_t GPUdb::getNumSymbols() const
{
	return metadata.numSymbols;
}

size_t GPUdb::getNumBlocks() const
{
	return metadata.numBlocks;
}

size_t GPUdb::getSequenceLength(size_t sequenceNum) const
{
	return seqNums[sequenceNum];
}

const char* GPUdb::getDescription(unsigned int index) const
{
	if(index>=descriptions.size())
		return NULL;
	return descriptions[index];
}


bool GPUdb::copyToGPU()
{
	//Prepare database for device access	
	if(cudaMalloc(&d_blob,blobSize)!=cudaSuccess)
		return false;	
	
	if(cudaMemcpy(d_blob,blob,blobSize,cudaMemcpyHostToDevice)!=cudaSuccess)
		return false;


	d_blockOffsets = (blockOffsetType*) d_blob;
	d_seqNums = (seqNumType*) ((char*) d_blockOffsets + metadata.numBlocks*sizeof(blockOffsetType));
	d_seqNums = (seqNumType*) ((char*) d_seqNums + metadata.alignmentPadding1);
	d_sequences = (seqType*) ((char*) d_seqNums + metadata.numBlocks*BLOCK_SIZE*sizeof(seqNumType));
	d_sequences = (seqType*) ((char*) d_sequences + metadata.alignmentPadding2);
	
	//Check alignment
	if((size_t) d_blockOffsets%256!=0)
		return false;
	if((size_t) d_seqNums%256!=0)
		return false;
	if((size_t) d_sequences%256!=0)
		return false;
	return true;
}

const GPUdb::blockOffsetType *GPUdb::get_d_BlockOffsets()
{
	return d_blockOffsets;
}

const GPUdb::seqNumType *GPUdb::get_d_SeqNums()
{
	return d_seqNums;
}

const GPUdb::seqType *GPUdb::get_d_Sequences()
{
	return d_sequences;
}

bool GPUdb::openOutFile(const char *fileName)
{
	outFile.open(fileName);
	if(!outFile.is_open())
		return false;
	return true;
}

/**
Write a sequence to file so it can be processed by a CPU tool such as SSearch
*/
bool GPUdb::writeToFile(size_t sequenceNum)
{
	if(strcmp(descriptions[sequenceNum],PADDING_SEQ_NAME) == 0)
		return true;
	outFile << '>' << descriptions[sequenceNum] << std::endl;

	size_t offset = sequenceOffsets[sequenceNum];

	seqType* ptr = sequences+offset;
			
	while(*ptr!=SUB_SEQUENCE_TERMINATOR&&*ptr!=SEQUENCE_GROUP_TERMINATOR)
	{
		for(size_t i=0;i<SUBBLOCK_SIZE;i++)
		{
			seqType val = ptr[i];
			if(val== ' ') //Subgroup padding
				break;
			outFile << FastaFile::AMINO_ACIDS[val];			
		}
		ptr+=GPUdb::BLOCK_SIZE*GPUdb::SUBBLOCK_SIZE;
	}
	outFile << '\n';
	return true;
}

void GPUdb::closeOutFile()
{
	outFile.close();
}
