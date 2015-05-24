/**
Class to read FASTA style substitution matrices from file and process them for GPU use.

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
#include "FastaMatrix.h"
#include "FastaFile.h"
#include <limits>
#include "GPUdb.h"
#include "main_cu.h"

FastaMatrix::FastaMatrix(): d_matrix(0), substArray(0), queryProfileArray(0), queryProfile(0)
{

}

FastaMatrix::~FastaMatrix()
{
	cudaFree(d_matrix);
	cudaFreeArray(substArray);
	cudaFreeArray(queryProfileArray);
	delete [] queryProfile;
}

/**
Load the matrix into a 2d associative map.
*/
bool FastaMatrix::load(const char *fileName)
{
	if(!fileName)
		return false;
	std::ifstream file;
	file.open(fileName);
	if(!file.is_open())
		return false;
				
	bool readHeader = false;
	std::string header;
	char row=' ';
	size_t column=0;
	while(1)
	{
		char c;
		file >> c;
		if(file.eof())
			break;

		//Skip comments
		if(c=='#')
		{			
			file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
			continue;
		}
		
		if(!readHeader) //Read row of amino acid letters
		{			
			if(header.find(c)!=std::string::npos) //Duplicate character: have gone past header into rows
				readHeader=true;
			else				
				header.append(1,c);
		}
		
		if(readHeader) //Interpret rows of values
		{
			if(header.find(c)!=std::string::npos) //Amino
			{
				row=c;
				column=0;
			}
			else //Value
			{				
				if(header.length()<column+1)
					return false;
				char columnName = header.at(column);
				column++;
				file.putback(c);
				char bla = file.peek();
				int val;
				file>>val;
				mapMatrix[row][columnName]=val;
			}
		}
			
	}
	
	file.close();
	
	return true;
}

bool FastaMatrix::copyToGPU()
{
	//Turn associative matrix into simple array ordered by alphabetical value
	for(size_t i=0;i<FastaFile::NUM_AMINO_ACIDS;i++)
	{
		for(size_t j=0;j<FastaFile::NUM_AMINO_ACIDS;j++)
		{
			matrix[i*FastaFile::NUM_AMINO_ACIDS+j]=mapMatrix[FastaFile::AMINO_ACIDS[i]][FastaFile::AMINO_ACIDS[j]];
		}
	}
	
	
	cudaMallocArray(&substArray,&getChannelDesc(),FastaFile::NUM_AMINO_ACIDS,FastaFile::NUM_AMINO_ACIDS);
	cudaMemcpyToArray(substArray,0,0,matrix,MATRIX_SIZE*sizeof(substType),cudaMemcpyHostToDevice);

	return true;
}



const cudaArray* FastaMatrix::getCudaArray()
{
	return substArray;
}

cudaChannelFormatDesc FastaMatrix::getChannelDesc()
{
	return cudaCreateChannelDesc(sizeof(substType)*8,0,0,0,cudaChannelFormatKindUnsigned);	
}

/**
Build a query profile
*/
bool FastaMatrix::createQueryProfile(FastaFile &query, bool dna)
{
	const char* ALPHABET;
	if(dna)
		ALPHABET = FastaFile::NUCLEOTIDES;
	else
		ALPHABET = FastaFile::AMINO_ACIDS;

	size_t seqLength = query.getSequenceLength(0);
	queryProfileLength = WHOLE_AMOUNT_OF(query.getSequenceLength(0),sizeof(queryType))*sizeof(queryType);
	char* seq = query.getSequence(0);
	size_t qpSize = queryProfileLength*FastaFile::NUM_AMINO_ACIDS;
	queryProfile = new substType[qpSize];
	if(!queryProfile)
		return false;

	for(size_t j=0;j<FastaFile::NUM_AMINO_ACIDS;j++)
	{
		char d = ALPHABET[j];
		for(size_t i=0;i<queryProfileLength;i++)
		{
			substType s;
			if(i>=seqLength) //If query sequence is too short, pad profile with zeroes
				s=0;
			else
			{
				char q = ALPHABET[seq[i]];
				s =mapMatrix[q][d];
			}
			queryProfile[j*queryProfileLength+i] = s;
		}
	}
	return true;
}

bool FastaMatrix::copyQueryProfileToGPU()
{
	if(cudaMallocArray(&queryProfileArray,&getQueryProfileChannelDesc(),WHOLE_AMOUNT_OF(queryProfileLength,sizeof(char4)),FastaFile::NUM_AMINO_ACIDS)!=cudaSuccess)
		return false;

	if(cudaMemcpyToArray(queryProfileArray,0,0,queryProfile,queryProfileLength*FastaFile::NUM_AMINO_ACIDS*sizeof(substType),cudaMemcpyHostToDevice)!=cudaSuccess)
		return false;

	return true;
}

const cudaArray* FastaMatrix::getQueryProfileCudaArray()
{
	return queryProfileArray;
}

cudaChannelFormatDesc FastaMatrix::getQueryProfileChannelDesc()
{
	return cudaCreateChannelDesc(sizeof(substType)*8,sizeof(substType)*8,sizeof(substType)*8,sizeof(substType)*8,cudaChannelFormatKindUnsigned);	
}
