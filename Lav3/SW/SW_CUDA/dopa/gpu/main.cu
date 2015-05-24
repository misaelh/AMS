/**
Meant for Cuda toolkit 3.1, 32 bit, SM 1.3, max registers set to 64

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
#include "main_cu.h"
#include "GPUdb.h"
#include "FastaFile.h"
#include "main.h"

//Adiga : Unix comptabile
//typedef unsigned char BYTE;

//__constant__ queryType c_query[63*1024/sizeof(queryType)]; /**< Query sequence in constant memory */
__constant__ int c_gapPenaltyTotal; //Gap + Extend penalties
__constant__ int c_gapExtendPenalty; //Extend penalty
__constant__ int c_queryLength; //Length of query sequence in chunks of 4

texture<char4,2> t_queryProfile;

/**
Align 4 query profile entries with a database residue.
*/
__device__  void alignResidues(scoreType &maxScore, scoreType4& left, int4& ixLeft, scoreType& top, scoreType& topLeft, int &IyTop, const char4 &substScore)
{	
	//q0
	ixLeft.x = max(0,max(left.x+c_gapPenaltyTotal,ixLeft.x+c_gapExtendPenalty)); //Max(0,...) here so IxColumn[] can be unsigned
	IyTop = max(top+c_gapPenaltyTotal,IyTop+c_gapExtendPenalty);
	int align = topLeft+substScore.x;
	topLeft=left.x;
	left.x = max(align,max(ixLeft.x,IyTop));
	
	//q1
	ixLeft.y = max(0,max(left.y+c_gapPenaltyTotal,ixLeft.y+c_gapExtendPenalty)); //Max(0,...) here so IxColumn[] can be unsigned
	IyTop = max(left.x+c_gapPenaltyTotal,IyTop+c_gapExtendPenalty);
	align = topLeft+substScore.y;
	topLeft=left.y;
	left.y = max(align,max(ixLeft.y,IyTop));

	//q2
	ixLeft.z = max(0,max(left.z+c_gapPenaltyTotal,ixLeft.z+c_gapExtendPenalty)); //Max(0,...) here so IxColumn[] can be unsigned
	IyTop = max(left.y+c_gapPenaltyTotal,IyTop+c_gapExtendPenalty);
	align = topLeft+substScore.z;
	topLeft=left.z;
	left.z = max(align,max(ixLeft.z,IyTop));

	//q3
	ixLeft.w = max(0,max(left.w+c_gapPenaltyTotal,ixLeft.w+c_gapExtendPenalty)); //Max(0,...) here so IxColumn[] can be unsigned
	IyTop = max(left.z+c_gapPenaltyTotal,IyTop+c_gapExtendPenalty);
	align = topLeft+substScore.w;	
	left.w = max(align,max(ixLeft.w,IyTop));

	topLeft=top; //The next column is to the right of this one, so current top left becomes new top
	top = left.w; //Set top value for next query chunk
	maxScore = max(left.x,max(left.y,max(left.z,max(left.w,maxScore)))); //Update max score
}			
			
/**
Align a database sequence subblock with the entire query sequence.
The loading/aligning with the query sequence in the 'inner' function as query sequence (constant) memory is much faster than the global memory in which the db resides.
*/
__device__ inline void alignWithQuery(const seqType8 &s, int column, TempData2* tempColumn, scoreType &maxScore)
{
		
		//Set the top related values to 0 as we're at the top of the matrix
		scoreType8 top = {0,0,0,0,0,0,0,0};
		scoreType topLeft = 0;
		int8 IyTop = {0,0,0,0,0,0,0,0};		

		char4 substScores; //Query profile scores
		scoreType4 left;
		int4 ixLeft;
		for(int j=0;j<c_queryLength;j++)
		{
			//Load first half of temporary column
			TempData2 t = tempColumn[0];
			left.x = mul24(column,t.a.F); 
			ixLeft.x = mul24(column, t.a.Ix);
			left.y = mul24(column,t.b.F);
			ixLeft.y = mul24(column, t.b.Ix);			

			//Load second half of temporary column
			t = tempColumn[gridDim.x*blockDim.x];
			left.z = mul24(column,t.a.F);
			ixLeft.z = mul24(column, t.a.Ix);
			left.w = mul24(column,t.b.F);
			ixLeft.w = mul24(column, t.b.Ix);

			int topLeftNext = left.w; //Save the top left cell value for the next loop interation

			//d0
			substScores = tex2D(t_queryProfile,j,s.a.x);
			alignResidues(maxScore, left, ixLeft, top.a.x, topLeft, IyTop.a.x, substScores);
			
			//d1
			substScores = tex2D(t_queryProfile,j,s.a.y);
			alignResidues(maxScore, left, ixLeft, top.a.y, topLeft, IyTop.a.y, substScores);

			//d2
			substScores = tex2D(t_queryProfile,j,s.a.z);
			alignResidues(maxScore, left, ixLeft, top.a.z, topLeft, IyTop.a.z, substScores);

			//d3
			substScores = tex2D(t_queryProfile,j,s.a.w);
			alignResidues(maxScore, left, ixLeft, top.a.w, topLeft, IyTop.a.w, substScores);

			//d4
			substScores = tex2D(t_queryProfile,j,s.b.x);
			alignResidues(maxScore, left, ixLeft, top.b.x, topLeft, IyTop.b.x, substScores);

			//d5
			substScores = tex2D(t_queryProfile,j,s.b.y);
			alignResidues(maxScore, left, ixLeft, top.b.y, topLeft, IyTop.b.y, substScores);

			//d6
			substScores = tex2D(t_queryProfile,j,s.b.z);
			alignResidues(maxScore, left, ixLeft, top.b.z, topLeft, IyTop.b.z, substScores);

			//d7
			substScores = tex2D(t_queryProfile,j,s.b.w);
			alignResidues(maxScore, left, ixLeft, top.b.w, topLeft, IyTop.b.w, substScores);

			topLeft = topLeftNext;
			
			//Save the two temporary column values
			t.a.F = left.x;
			t.a.Ix = ixLeft.x;
			t.b.F = left.y;
			t.b.Ix = ixLeft.y;
			tempColumn[0]=t;
			tempColumn+=gridDim.x*blockDim.x;
			t.a.F = left.z;
			t.a.Ix = ixLeft.z;
			t.b.F = left.w;
			t.b.Ix = ixLeft.w;
			tempColumn[0]=t;

			tempColumn+=gridDim.x*blockDim.x;			
		}
}


/**
Align the database sequence subblocks with the query sequence until a terminating subblock is encountered.
*/
__device__ void align(const GPUdb::seqType* sequence, TempData2* const tempColumn, GPUdb::seqSizeType& seqNum, scoreType* scores)
{
	scoreType maxScore=0;

	seqType8 s;
	int column = 0; //Column = 0 means that alignment function will use 0 for 'left' values as there's no left column to read from
	s = *(seqType8*)sequence;
	while(s.a.x!=' ') //Until terminating subblock
	{		
		if(s.a.x=='#') //Subblock signifying concatenated sequences
		{ 
			scores[seqNum] = maxScore;
			seqNum++;
			column=maxScore=0;
		}

		alignWithQuery(s,column,tempColumn,maxScore);
			
		column=1;
		sequence += GPUdb::BLOCK_SIZE*GPUdb::SUBBLOCK_SIZE;	
		s = *(seqType8*)sequence;
	}
	
	scores[seqNum] = maxScore; //Set score for sequence
	return;

}

/**
Main kernel function
*/
__global__ void smithWaterman(int numGroups,scoreType* scores, const GPUdb::blockOffsetType* blockOffsets,const GPUdb::seqNumType* seqNums, const GPUdb::seqType* sequences,  TempData2* const tempColumns)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int groupNum = idx;

	TempData2* const tempColumn = &tempColumns[idx]; //Access temporary values interlaced

	while(groupNum<numGroups)
	{
		GPUdb::seqNumType seqNum=seqNums[groupNum];	//Get sequence number of first sequence in group
	
		//Calculate memory address of sequence group
		int seqBlock = groupNum >> GPUdb::LOG2_BLOCK_SIZE;
		int groupNumInBlock = groupNum & (GPUdb::BLOCK_SIZE-1); //equals groupNum % GPUdb::BLOCK_SIZE
		int groupOffset = blockOffsets[seqBlock]+__umul24(groupNumInBlock,GPUdb::SUBBLOCK_SIZE);
		const GPUdb::seqType* group = &sequences[groupOffset];
	
				
		align(group,tempColumn,seqNum,scores); //Perform alignment
		groupNum+= gridDim.x*blockDim.x;
	
	}

}

// main routine that executes on the host
bool launchSW(scoreType** scores, void* query, size_t queryLength, GPUdb& db, FastaMatrix& substitutionMatrix, int gapPenalty, int gapExtendPenalty, double& time)
{
	/*
	//Prepare substitution matrix
	if(!substitutionMatrix.copyToGPU())
	{
		puts("Error uploading substitution matrix.");
		return false;
	}
	t_substMatrix.addressMode[0] = cudaAddressModeWrap;
	t_substMatrix.addressMode[1] = cudaAddressModeWrap;
	t_substMatrix.filterMode = cudaFilterModePoint;
	t_substMatrix.normalized = false;
	cudaBindTextureToArray(t_substMatrix,substitutionMatrix.getCudaArray(),substitutionMatrix.getChannelDesc());	
	*/

	//Prepare substitution matrix
	
	if(!substitutionMatrix.copyQueryProfileToGPU())
	{
		puts("Error uploading query profile.");
		return false;
	}
	t_queryProfile.addressMode[0] = cudaAddressModeWrap;
	t_queryProfile.addressMode[1] = cudaAddressModeWrap;
	t_queryProfile.filterMode = cudaFilterModePoint;
	t_queryProfile.normalized = false;
	cudaBindTextureToArray(t_queryProfile,substitutionMatrix.getQueryProfileCudaArray(),substitutionMatrix.getQueryProfileChannelDesc());	

	//Prepare database
	if(!db.copyToGPU())
	{
		puts("Error uploading database.");
		return false;
	}

	//Prepare query
/*	if(cudaMemcpyToSymbol(c_query,query,queryLength,0,cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		puts("Error copying query sequence.");
		return false;
	}*/
	printf("111\n");
	printf("\n %d \n" ,queryLength);
		printf("\n %d \n" ,sizeof(queryType));
	size_t queryLengthInChunks = WHOLE_AMOUNT_OF(queryLength,sizeof(queryType));
	size_t queryLengthDiv2InChunks = WHOLE_AMOUNT_OF(queryLength/2,sizeof(queryType));
	//if(cudaMemcpyToSymbol(c_queryLength,&queryLengthInChunks,sizeof(queryLengthInChunks),0,cudaMemcpyHostToDevice)!=cudaSuccess)
		printf("222  %d   %d \n",queryLengthInChunks,c_queryLength);
   // queryLengthInChunks=abs(queryLengthInChunks);
	if(cudaMemcpyToSymbol(c_queryLength,&queryLengthInChunks,sizeof(int),0,cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		puts("Error copying query sequence length.");
		return false;
	}

	//Prepare penalties
	int gapPenaltyTotal = gapPenalty + gapExtendPenalty;
	if(cudaMemcpyToSymbol(c_gapPenaltyTotal,&gapPenaltyTotal,sizeof(gapPenaltyTotal),0,cudaMemcpyHostToDevice)!=cudaSuccess
	| cudaMemcpyToSymbol(c_gapExtendPenalty,&gapExtendPenalty,sizeof(gapExtendPenalty),0,cudaMemcpyHostToDevice)!=cudaSuccess)
	{
		puts("Error copying penalties.");
		return false;
	}

	//Prepare score array
	size_t scoreArraySize = sizeof(scoreType)*db.getNumSequences();
	if(cudaMallocHost(scores,scoreArraySize)!=cudaSuccess)
	{
		puts("Error allocating host score array.");
		return false;
	}
	
	scoreType* d_scores;
	if(cudaMalloc(&d_scores,scoreArraySize)!=cudaSuccess)
	{
		puts("Error allocating device score array: database too large?.");
		return false;
	}
	cudaMemset(d_scores,-1,scoreArraySize); //Set scores to -1 so we can check if they were actually all written by the kernel.
	

	int matrixSize = queryLengthDiv2InChunks*sizeof(queryType)*sizeof(TempData2); //Size of temporary storage for one thread

	//Determine launch configuration
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props,0);
	size_t free, total;
	cudaMemGetInfo(&free,&total);
		
	if(db.getNumSequences() < 16)
	{
		puts("Database does not contain enough sequences to have a full thread block.");
		return false;
	}
	
		
	int blocks = props.multiProcessorCount*4;
	int threadsPerBlock = props.maxThreadsPerBlock/8; //Blocksize should be a multiple of 32 threads

	int blocksPerhw = (int)ceil((double)db.getNumBlocks()/(double)(blocks*threadsPerBlock/16));



	//threadsPerBlock=1;blocks=32;
	printf("Using %d blocks of %d threads: %d threads for %d sequences in %d blocks.\n",blocks,threadsPerBlock,blocks*threadsPerBlock,db.getNumSequences(),db.getNumBlocks());
	printf("Processing %d blocks per half warp.\n",blocksPerhw);
	//printf("Processing %d sequences per thread.\n",symbolsPerThread);

	//Prepare temporary score matrices: one F and one Ix column per thread
	TempData2* d_tempColumns;
	matrixSize *= blocks*threadsPerBlock;
	
	if(cudaMalloc(&d_tempColumns,matrixSize)!=cudaSuccess)
	{
		puts("Error allocating temporary matrix: too many threads for sequence size?");
		return false;
	}
	/*if(cudaMemset(d_tempColumns,0,matrixSize)!=cudaSuccess)
	{
		puts("Error allocating temporary matrix.");
		return false;
	}*/

	//Run kernel
	fflush(stdout);
	puts("Running...");

	
	clock_t start = clock();
	smithWaterman <<<blocks,threadsPerBlock>>>(db.getNumBlocks()*GPUdb::BLOCK_SIZE,d_scores,db.get_d_BlockOffsets(),  db.get_d_SeqNums(),db.get_d_Sequences(), d_tempColumns);	

	if(cudaThreadSynchronize()!=cudaSuccess)
		return false;

	clock_t stop = clock();
	time=(stop-start)/(double)CLOCKS_PER_SEC;

	//Get scores
	cudaMemcpy((void*) *scores,d_scores,scoreArraySize,cudaMemcpyDeviceToHost);
	

	cudaFree(d_tempColumns);	
	cudaFree(d_scores);
	return true;
}
