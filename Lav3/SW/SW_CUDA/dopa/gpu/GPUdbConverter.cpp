/**
Class to convert FASTA databases to GPU format.

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

===

Format:
int32 Number of sequences
int32 Number of symbols
int32 Number of blocks
int32 Bytes of padding between block offsets and sequence numbers
int32 Bytes of padding between sequence lenghts and sequences
int32 offset to each sequence (for easier exporting)

(the next items are uploaded to the GPU and are known as the 'blob')

for each block: int32 offset to block
padding to align next part
for each sequence group int32 first sequence of group
padding to align next part
for each block: chunks of SUBBLOCK_SIZE characters of each sequence interlaced with eachother. There are BLOCK_SIZE sequence groups in each block.

A sequence group consists of concatenated subsequences with a subsequence-terminator subblock between them. A sequence group in turn is terminated by a sequence group terminator subblock.
To prevent wasted space and GPU processing time, sequences are concatenated so that each sequence group in a block has about the same length.
Furthermore, it was discovered that a speedup could be gained by having each group have the same size. 
So each group will be filled with (concatenated) sequences until it's about the length of the single longest sequence in the database.

======

Suppose we have a sequence A of length 16, and B,C,D of length 6 and E,F of length 4 and G of length 3. Block size is 3 sequence groups. Subblock size is 2 characters.
# = subblock terminator, space = group terminator and subblock padding, 0 = empty file space. The database would then look like this:

block 0:

AABBDD
AABBDD
AABBDD
AA####
AACCEE
AACCEE
AACC      
AA  00
  0000

block 1:
FFGG  
FFG 00
    00
*/

#include "GPUdbConverter.h"
#include <fstream>
#include <algorithm>
#include <string.h>
FastaFile::FastaRecord GPUdbConverter::padding;

GPUdbConverter::GPUdbConverter(): FastaFile(), blob(0), sequenceOffsets(0), blobSize(0), sequenceNumbers(0)
{

}

GPUdbConverter::~GPUdbConverter()
{
	delete [] blob;
	delete [] sequenceOffsets;
	delete [] sequenceNumbers;
}

/**
Conversion has two steps: rearranging sequences and writing to file.

The rearrangement step sorts sequences by length (longest first) and makes a copy of the sequence list. Sequences are then taken from this copy and put into blocks until no are left.
The final block is padded to a full block with bogus, empty, sequences.

The length of a block is determined by the length of its longest (first) sequence. To decrease GPU idle time,
leftover space resulting from block sequences being shorter than the block length is filled with as long as possible sequences that are left.
Together this original sequence and the ones added to lengthen it are called a 'sequence group'.

Writing to file is done in the format outlined above. Sequences are padded to a full amount of subblocks, and after each sequence a terminator subblock is inserted
so the GPU knows to stop reading. Different terminators are used between sequences in a sequence group and after the last sequence of the group. 

*/
bool GPUdbConverter::convert()
{
	stats.numPadding = 0;
	stats.numFillSequences = 0;

	arrangeSequences();

	sequenceOffsets = new size_t[totalNumSequences];
	sequenceNumbers = new size_t[blocks.size()*GPUdb::BLOCK_SIZE];

	createSequenceBlob();

	return true;
}

void GPUdbConverter::arrangeSequences()
{
	//Sort by length
	std::sort(records.begin(),records.end(),&FastaFile::comparisonFunc);
	//dump();


	//Make copy of the pointers
	std::vector<FastaRecord*> records2;
	records2.resize(records.size());
	for(size_t i=0;i<records.size();i++)
	{
		//if(i>records.size()-17)
		//	printf("%d\n",records[i].length);
		records2[i] = &records[i];
	}


	size_t maxLength = WHOLE_AMOUNT_OF(records2.back()->length,GPUdb::SUBBLOCK_SIZE);
	while(records2.size() > 0)
	{
		//Set length of each block to length of longest sequence
		Block b;
		b.length = maxLength;
		//b.length = WHOLE_AMOUNT_OF(records2.back()->length,GPUdb::SUBBLOCK_SIZE);
		
		//Fill slots in block
		for(size_t i=0;i<GPUdb::BLOCK_SIZE;i++)
		{
			if(records2.size()!=0)
			{
				//Add longest leftover sequence
				b.sequenceGroups[i].push_back(records2.back());
				int delta = b.length - WHOLE_AMOUNT_OF(records2.back()->length,GPUdb::SUBBLOCK_SIZE);
				records2.pop_back();
				
				//Fill blank space left by size difference between longest sequence and this sequence with other sequences
				//Approaches to try: fill as much as possible, fill just once, fill only with sequences of some minimum length, fill only starting at end of one-but-longest sequence of group
				while(delta > 1 && records2.size() > 0)
				{
					delta--;  //-1 because a separator block must be inserted
					//Find other sequences to fill up empty space
					FastaFile::FastaRecord d;
					d.length = delta*GPUdb::SUBBLOCK_SIZE;
					std::vector<FastaRecord*>::iterator fillItem;
					fillItem = std::lower_bound(records2.begin(),records2.end(),&d,&FastaFile::comparisonFuncPtr);
					if(fillItem == records2.begin()) //No suitable sequence found
						break;
					fillItem--;
					b.sequenceGroups[i].push_back(*fillItem);
					delta -= WHOLE_AMOUNT_OF((*fillItem)->length,GPUdb::SUBBLOCK_SIZE);
					records2.erase(fillItem);
					stats.numFillSequences++;					
				}
			}
			else //No sequences left
				break;	
		}


		
		b.length += 1; //Terminator block
		blocks.push_back(b);
	}
		

	//Make the last block a whole block
	padding.length = 0;
	padding.description = PADDING_SEQ_NAME;
	totalNumSequences = records.size();
	for(size_t i=0;i<GPUdb::BLOCK_SIZE;i++)
	{
		if(blocks.back().sequenceGroups[i].size()==0)
		{
			blocks.back().sequenceGroups[i].push_back(&padding);
			totalNumSequences++;
		}
	}
}

/**
Create the blob of interlaced sequences the GPU will use
*/
void GPUdbConverter::createSequenceBlob()
{
	blobSize = 0;
	for(size_t i=0;i<blocks.size();i++)
	{
		blobSize += blocks[i].length;
	}
	blobSize *= GPUdb::SUBBLOCK_SIZE*GPUdb::BLOCK_SIZE;
	blob = new char[blobSize];
	memset(blob,0,blobSize);
	

	char* ptr = blob;
	for(size_t i=0;i<blocks.size();i++)
	{
		writeBlock(ptr, i);
		ptr+=blocks[i].length*GPUdb::BLOCK_SIZE*GPUdb::SUBBLOCK_SIZE*sizeof(GPUdb::seqType);
	}

}

/**
Write a single block to the blob
*/
void GPUdbConverter::writeBlock(char* ptr, size_t block)
{
	static size_t groupCounter;
	for(size_t sequence = 0; sequence < GPUdb::BLOCK_SIZE; sequence++)
	{	
			writeSequenceGroup(ptr, blocks[block].sequenceGroups[sequence],groupCounter);
			ptr+=GPUdb::SUBBLOCK_SIZE*sizeof(GPUdb::seqType);
			groupCounter++;
	}	
}

/**
Write a sequence group, adding terminating subchunks where necessary
*/
void GPUdbConverter::writeSequenceGroup(char* ptr, std::vector<FastaRecord*> &group, size_t groupNum)
{
	static size_t sequenceCounter;

	sequenceNumbers[groupNum] = sequenceCounter;
	//For each sequence in the group
	for(size_t sequence = 0; sequence < group.size(); sequence++)
	{
		//Calculate offset
		sequenceOffsets[sequenceCounter] = ptr - blob;		
		
		sequenceCounter++;

		//Write subblocks
		for(size_t subBlock = 0; subBlock < WHOLE_AMOUNT_OF(group[sequence]->length,GPUdb::SUBBLOCK_SIZE); subBlock++)
		{
			for(size_t symbol = 0; symbol < GPUdb::SUBBLOCK_SIZE; symbol++)
			{
				size_t totalSymbol = symbol + subBlock * GPUdb::SUBBLOCK_SIZE;
				if(totalSymbol >= group[sequence]->length) //Pad to full amount of subblocks
				{
					ptr[symbol] = ' ';
					stats.numPadding++;
				}
				else
					ptr[symbol] = group[sequence]->sequence[totalSymbol];
			}
			ptr+=GPUdb::BLOCK_SIZE*GPUdb::SUBBLOCK_SIZE*sizeof(GPUdb::seqType); //Move to proper position for interlacing
		}
		
		//Write terminating subblock
		for(size_t symbol = 0; symbol < GPUdb::SUBBLOCK_SIZE; symbol++)
		{
			if(sequence < group.size()-1) //Not the final sequence
				ptr[symbol] = GPUdb::SUB_SEQUENCE_TERMINATOR;
			else //Final sequence of group
				ptr[symbol] = GPUdb::SEQUENCE_GROUP_TERMINATOR; 
		}
		ptr+=GPUdb::BLOCK_SIZE*GPUdb::SUBBLOCK_SIZE*sizeof(GPUdb::seqType); //Move to proper position for interlacing
	}

	
}

/**
Write converted file to disk
*/
bool GPUdbConverter::write(const char* fileName)
{	
	std::ofstream file;
	file.open(fileName,std::ios::binary);
	if(!file.is_open())
		return false;

	

	//Write number of sequences
	size_t s = totalNumSequences;
	file.write((const char*)&s,sizeof(s));
	
	//Write total number of symbols
	s = numSymbols;
	file.write((const char*)&s,sizeof(s));

	//Write total number of blocks
	s = blocks.size();
	file.write((const char*)&s,sizeof(s));

	//Write alignment padding amount 1
	size_t padding1;
	if(GPUdb::ALIGNMENT==0)
		padding1 = 0;
	else
		padding1 = GPUdb::ALIGNMENT-blocks.size()*sizeof(GPUdb::blockOffsetType)%GPUdb::ALIGNMENT;
	file.write((const char*)&padding1,sizeof(padding1));

	//Write alignment padding amount 2
	size_t padding2;
	if(GPUdb::ALIGNMENT==0)
		padding2 = 0;
	else
		padding2 = GPUdb::ALIGNMENT-blocks.size()*GPUdb::BLOCK_SIZE*sizeof(GPUdb::seqNumType)%GPUdb::ALIGNMENT;
	file.write((const char*)&padding2,sizeof(padding2));

	//Write sequence offsets
	file.write((const char*)sequenceOffsets,totalNumSequences*sizeof(size_t));
	
	
	//Write block offsets
	GPUdb::blockOffsetType o=0;
	for(size_t i=0;i<blocks.size();i++)
	{				
		file.write((const char*)&o,sizeof(o));
		o+=GPUdb::BLOCK_SIZE*blocks[i].length*GPUdb::SUBBLOCK_SIZE*sizeof(GPUdb::seqType);
	}

	//Write alignment padding
	for(size_t i=0;i<padding1;i++)
		file << '0';
	
	//Write number of first sequence in group
	for(size_t i=0;i<GPUdb::BLOCK_SIZE*blocks.size();i++)
	{		
		
		GPUdb::seqNumType n = sequenceNumbers[i];
		file.write((const char*)&n,sizeof(n));
	}	

	//Write alignment padding	
	for(size_t i=0;i<padding2;i++)
		file << '0';


	//Write blob
	file.write((const char*)blob,blobSize);


	printf("%d blocks\n",blocks.size());
	printf("%d sequences used to fill gaps\n",stats.numFillSequences);
	printf("%d bytes of alignment padding inserted.\n",padding1+padding2);
	printf("%d bytes of padding inserted.\n",stats.numPadding);
	printf("%f new vs original size ratio.\n",blobSize/(float)numSymbols);
	
	file.close();

	return true;
}


bool GPUdbConverter::writeDescriptions(const char *fileName)
{
	std::ofstream file;
	file.open(fileName);
	if(!file.is_open())
		return false;

	for(size_t i=0;i<blocks.size();i++)
	{
		for(size_t j=0;j<GPUdb::BLOCK_SIZE;j++)
		{
			for(size_t k=0;k<blocks[i].sequenceGroups[j].size();k++)
			{
				file << blocks[i].sequenceGroups[j][k]->description << '\n';
			}
		}		
	}

	file.close();
	return true;
}
