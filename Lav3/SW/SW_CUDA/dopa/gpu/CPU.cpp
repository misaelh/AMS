/**
CPU implementation to check GPU version with, not necessarily up to date with GPU one.

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
#include "CPU.h"
#include <string.h>

static CPU::cpuCellType** FMatrix;
static CPU::cpuCellType** IxMatrix;
static CPU::cpuCellType** IyMatrix;

int CPU::subst(char c1,char c2)
{
	if(c1=='X' && c2== 'X')
		return 0;
	if( c1==c2)
		return 4;
	return -10;
}

CPU::cpuCellType CPU::align(const char* query, size_t queryLen, const char* db, size_t dbLen, FastaMatrix& substitutionMatrix, int gapPenalty, int gapExtendPenalty)
{
	CPU::cpuCellType maxScore=0;
	for(size_t i=1;i<dbLen+1;i++)
	{
		char d=db[i-1];
		for(size_t j=1;j<queryLen+1;j++)
		{
			
			char q=query[j-1];
			int score;

			int left = FMatrix[j][i-1];
			int top = FMatrix[j-1][i];
			int topLeft = FMatrix[j-1][i-1];

			

			int align=topLeft+subst(d,q);
			score=std::max<int>(align,0);	
	
			int newIx=std::max<int>(left+gapPenalty+gapExtendPenalty,IxMatrix[j][i-1]+gapExtendPenalty);
			score = std::max<int>(score,newIx);
		
			int newIy=std::max<int>(top+gapPenalty+gapExtendPenalty,IyMatrix[j-1][i]+gapExtendPenalty);
			score = std::max<int>(score,newIy);	
			
			maxScore=std::max<int>(score,maxScore);

			FMatrix[j][i]=score;
			IxMatrix[j][i]=newIx;
			IyMatrix[j][i]=newIy;
			
			

		}
	}

	return maxScore;
}


CPU::cpuCellType CPU::align2(const char* query, size_t queryLen, const char* db, size_t dbLen, FastaMatrix& substitutionMatrix, int gapPenalty, int gapExtendPenalty)
{
	CPU::cpuCellType maxScore=0;
	CPU::cpuCellType *FColumn = FMatrix[0];
	CPU::cpuCellType *IxColumn = IxMatrix[0];
	for(size_t i=1;i<dbLen+1;i++)
	{
		char d=db[i-1];

		int top=0;
		int left=0;
		int topLeft=0;
		int topIy=0;
		for(size_t j=1;j<queryLen+1;j++)
		{
			
			char q=query[j-1];
			int score;

			left = FColumn[j];

			int align=topLeft+subst(d,q);
			score=std::max<int>(align,0);	
	
			int newIx=std::max<int>(left+gapPenalty+gapExtendPenalty,IxColumn[j]+gapExtendPenalty);
			score = std::max<int>(score,newIx);
		
			int newIy=std::max<int>(top+gapPenalty+gapExtendPenalty,topIy+gapExtendPenalty);
			score = std::max<int>(score,newIy);	
			
			maxScore=std::max<int>(score,maxScore);

			FColumn[j]=score;
			IxColumn[j]=newIx;
			top = score;
			topLeft = left;
			topIy = newIy;
		}
	}

	return maxScore;
}


bool CPU::launchSW(FastaFile& query, FastaFile &db, FastaMatrix& substitutionMatrix, int gapPenalty, int gapExtendPenalty)
{

	
	for(size_t i=0;i<db.getNumSequences();i++)
	{
		size_t matSize = std::max<int>(db.getSequenceLength(i)+1,query.getSequenceLength(0)+1);
		FMatrix = new cpuCellType*[matSize];
		IxMatrix = new cpuCellType*[matSize];
		IyMatrix = new cpuCellType*[matSize];
		for(size_t j=0;j<matSize;j++)
		{
		
			FMatrix[j] = new cpuCellType[matSize];
			memset(FMatrix[j],0,matSize*sizeof(CPU::cpuCellType));
			IxMatrix[j] = new cpuCellType[matSize];
			memset(IxMatrix[j],0,matSize*sizeof(CPU::cpuCellType));
			IyMatrix[j] = new cpuCellType[matSize];
			memset(IxMatrix[j],0,matSize*sizeof(CPU::cpuCellType));
		}
		

		
		int score = align(query.getSequence(0),query.getSequenceLength(0),db.getSequence(i),db.getSequenceLength(i),substitutionMatrix,gapPenalty,gapExtendPenalty);
		int score2 = align2(query.getSequence(0),query.getSequenceLength(0),db.getSequence(i),db.getSequenceLength(i),substitutionMatrix,gapPenalty,gapExtendPenalty);

		if(score2!=score)
		{
			printf("Alignment method difference: %d %d\n", score,score2);
		}
		for(size_t j=0;j<matSize;j++)
		{
			delete [] FMatrix[j];
			delete [] IxMatrix[j];
			delete [] IyMatrix[j];
		}

		delete [] FMatrix;
		delete [] IxMatrix;
		delete [] IyMatrix;
		printf("%3d. %-50.50s\t SCORE: %d\n",i,db.getDescription(i),score);
	}
	return true;
}
