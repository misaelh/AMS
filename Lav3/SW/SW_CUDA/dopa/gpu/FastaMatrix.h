#pragma once
#include "FastaFile.h"
#include <map>

// Adiga : Include two struct definitions
//struct cudaArray;
//struct cudaChannelFormatDesc;

class FastaMatrix
{
public:
	static const int MATRIX_SIZE = FastaFile::NUM_AMINO_ACIDS*FastaFile::NUM_AMINO_ACIDS;
	typedef char substType;

private:
	substType matrix[MATRIX_SIZE];
	substType* d_matrix;
	substType* queryProfile;
	cudaArray* substArray;
	cudaArray* queryProfileArray;
	size_t queryProfileLength;
	std::map< char,std::map<char,int> > mapMatrix; /**< Map for convenient host-side storage of matrix */
	
	
public:
	FastaMatrix();
	~FastaMatrix();
	bool load(const char *fileName);
	
	
	//const substType *get_d_Matrix();
	//Matrix functions
	bool copyToGPU();
	const cudaArray* getCudaArray();
	cudaChannelFormatDesc getChannelDesc();

	//Query profile functions
	bool createQueryProfile(FastaFile &query, bool dna = false);
	bool copyQueryProfileToGPU();
	const cudaArray* getQueryProfileCudaArray();
	cudaChannelFormatDesc getQueryProfileChannelDesc();
	
};
