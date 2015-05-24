/**
Test database generator
*/

#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <fstream>
//#include "../gpu/FastaFile.h"
#include "/home/et4381-5/Sriram/gasw_windows/source/gpu/FastaFile.h"


static std::ofstream file;

void writeSequences(int num, int minLength, int maxLength)
{
	for(int i=0;i<num;i++)
	{
		file << ">test|test|Test sequence " << i << std::endl;
		int length = minLength;
		if(minLength!=maxLength)
			length += rand()%(maxLength-minLength);
		for(int j=0;j<length;j++)
		{
			int residue = rand()%(FastaFile::NUM_AMINO_ACIDS-1);
			file << FastaFile::AMINO_ACIDS[residue];
		}
		file << std::endl;

	}
}

int main(int argc, char* argv[])
{
	srand((unsigned int)time(0));
	
	file.open("test.fasta");
	
	//writeSequences(510000,352,352); //Sequences with length of average Swiss-Prot sequence
//	writeSequences(100,50,2000);
			//writeSequences(383*16,3520,3520);
	//writeSequences(240*1000,352,352); //Benchmark
	writeSequences(480*16*32,352,352); //Benchmark 2
	
	
	
	puts("Done.");
	getchar();
	file.close();
}
