#pragma once

#include "FastaFile.h"
#include "FastaMatrix.h"
class CPU
{
	
public:
	typedef int cpuCellType;
	static int subst(char c1, char c2);
	static cpuCellType align(const char* query, size_t queryLen, const char* db, size_t dbLen, FastaMatrix& substitutionMatrix, int gapPenalty, int gapExtendPenalty);
	static cpuCellType align2(const char* query, size_t queryLen, const char* db, size_t dbLen, FastaMatrix& substitutionMatrix, int gapPenalty, int gapExtendPenalty);
	static bool launchSW(FastaFile& query, FastaFile &db, FastaMatrix& substitutionMatrix, int gapPenalty, int gapExtendPenalty);
};
