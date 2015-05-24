/**
Class to handle loading of FASTA format sequence (database) files.

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

#include "FastaFile.h"
#include <fstream>
#include <string.h>
const char* FastaFile::NUCLEOTIDES = "ACGTURYKMSWBDHVNX";
const char FastaFile::AMINO_ACIDS[FastaFile::NUM_AMINO_ACIDS+1] = "ABCDEFGHIKLMNPQRSTVWYZX";
const char* FastaFile::UNSUPPORTED_LETTERS = "UO*"; // http://faculty.virginia.edu/wrpearson/fasta/fasta36/
const char* FastaFile::UNSUPPORTED_LETTERS_REPLACEMENTS = "CKX";

FastaFile::FastaFile(): buffer(0), numSymbols(0)
{
	
}

FastaFile::~FastaFile()
{
	delete [] buffer;
}

/**
Load a FASTA file by reading it into memory and indexing pointers to its records and descriptions
*/
bool FastaFile::load(const char *fileName, bool dna)
{
printf("1\n");
	const char* ALPHABET;
	if(dna)
		ALPHABET = NUCLEOTIDES;
	else
		ALPHABET = AMINO_ACIDS;

	std::ifstream file;
	file.open(fileName,std::ios::binary);
	if(!file.is_open())
		return false;
	//Get size
	unsigned int fsize;
	file.seekg(0, std::ios::end);
	fsize = file.tellg();

	if(fsize < 1) //Empty file
	{
		file.close();
		return false;
	}

	//Read file into memory
	buffer = new char[fsize+1];
	file.seekg(0);
	file.read(buffer,fsize);
	buffer[fsize]=NULL;
	file.close();
	if(file.bad())
		return false;

	//Process records
	FastaRecord r;
	r.length = 0;
	char* context;
	char* tokStr=buffer+1;//Skip initial '>'
	while(1)
	{
		r.description=strtok(tokStr,"\n");		
		r.sequence=strtok(NULL,">");
		if(!r.description || !r.sequence)
			break;
		records.push_back(r);
		tokStr=NULL;
	}	

	//Strip unwanted characters
	for(size_t i=0;i<records.size();i++)
	{

		char* badChar;
		//Strip newlines from description
		while(badChar=strpbrk(records[i].description,"\r\n"))
			*badChar='\0';

		int copyAmt = 0;

		//For each bad character, increase the shift amount so we don't have to perform any superfluous copies
		size_t recLen = strlen(records[i].sequence)+1;
		for(char* c=records[i].sequence;c<records[i].sequence+recLen;c++) //+1 so NULL gets copied
		{
			*c=toupper(*c);
			const char* badResidue;
			if(!dna)
			{
				if(*c!=NULL&&(badResidue=strchr(UNSUPPORTED_LETTERS,*c))!=NULL) //Replace unsupported symbols by replacements
				{
					*c=UNSUPPORTED_LETTERS_REPLACEMENTS[badResidue-UNSUPPORTED_LETTERS];
				}
			}

			const char* residueIndex;

			if((residueIndex=strchr(ALPHABET,*c))==NULL) //Invalid character, skip it
			{
				copyAmt--;
				if(*c!='\n' && *c!='\r' && *c != ' ') //Usually the unsupported characters should only be whitespace and newlines
				{
					printf("Deleted unknown character %c\n",*c);
				}
			}

			else
			{
				if(*c!=NULL)
				{
					records[i].length++;
					numSymbols++;
					*c=residueIndex-ALPHABET; //Replace symbol with its alphabetic index
				}
				if(copyAmt!=0)
					*(c+copyAmt)=*c;
			}
		}
	}
	return true;
}

/**
Comparison function for sorting
*/
bool FastaFile::comparisonFunc(const FastaRecord &r1, const FastaRecord &r2)
{
	return (r1.length<r2.length);
}
bool FastaFile::comparisonFuncPtr(const FastaRecord *r1, const FastaRecord *r2)
{
	return (r1->length<r2->length);
}


size_t FastaFile::getNumSequences() const
{
	return records.size();
}

size_t FastaFile::getNumSymbols() const
{
	return numSymbols;
}

size_t FastaFile::getSequenceLength(size_t sequenceNum) const
{
	if(sequenceNum >= records.size())
		return 0;

	return records[sequenceNum].length;
}

char* FastaFile::getSequence(size_t sequenceNum) const
{
	if(sequenceNum >= records.size())
		return NULL;
	
	return records[sequenceNum].sequence;

}

const char* FastaFile::getDescription(size_t sequenceNum) const
{
	if(sequenceNum >= records.size())
		return NULL;
	
	return records[sequenceNum].description;
}

void FastaFile::dump()
{
	std::ofstream outFile;
	outFile.open("dump.fasta");

	for(size_t i=0;i<records.size();i++)
	{
		outFile<< records[i].description << '\n';
		for(size_t j=0;j<records[i].length;j++)
			outFile<<FastaFile::AMINO_ACIDS[records[i].sequence[j]];
		outFile<<'\n';
	}
	outFile.close();
}
