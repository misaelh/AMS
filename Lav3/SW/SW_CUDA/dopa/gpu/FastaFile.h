#pragma once
#include <vector>
#include <cstddef>

class FastaFile
{
private:
	char* buffer; /**< Buffer that is whole file loaded into memory */
protected:
	struct FastaRecord
	{
		char* sequence;
		char* description;
		size_t length;
	};	
	std::vector<FastaRecord> records; /**< Records index into buffer */
	static bool comparisonFunc(const FastaRecord&,const FastaRecord&);
	static bool comparisonFuncPtr(const FastaRecord*,const FastaRecord*);
	unsigned int numSymbols;

public:
	static const int NUM_AMINO_ACIDS = 24;
	static const char* NUCLEOTIDES;
	static const char AMINO_ACIDS[NUM_AMINO_ACIDS+1];
	static const char* UNSUPPORTED_LETTERS;
	static const char* UNSUPPORTED_LETTERS_REPLACEMENTS;
	
	FastaFile();
	virtual ~FastaFile();
	bool load(const char* fileName, bool dna=false);
	size_t getNumSequences() const;
	size_t getNumSymbols() const;
	size_t getSequenceLength(size_t sequenceNum) const;
	char* getSequence(size_t sequenceNum) const;
	const char* getDescription(size_t sequenceNum) const;

	void dump();
};
