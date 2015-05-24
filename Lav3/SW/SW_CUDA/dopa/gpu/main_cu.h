#pragma once
#include "GPUdb.h"
#include "FastaMatrix.h"

//#if defined(__CUDACC__) // NVCC
//   #define MY_ALIGN(n) __align__(n)
//#else
//  #define MY_ALIGN(n) __attribute__((aligned(n)))
//#endif
//#elif defined(__GNUC__) // GCC
//  #define MY_ALIGN(n) __attribute__((aligned(n)))
//#elif defined(_MSC_VER) // MSVC
//  #define MY_ALIGN(n) __declspec(align(n))
//#else
//  #error "Please provide a definition for MY_ALIGN macro for your host compiler!"
//#endif

struct __align__(32) int8
{
	int4 a;
	int4 b;
};

struct __align__(64) int16
{
	int4 a;
	int4 b;
	int4 c;
	int4 d;
};

struct __align__(8) char8
{
	char4 a;
	char4 b;
};

typedef char2 seqType2;
typedef char4 seqType4;
typedef unsigned short scoreType;
typedef ushort2 scoreType2;
typedef ushort4 scoreType4;
typedef char4 queryType;

struct __align__(8) seqType8
{
	seqType4 a;
	seqType4 b;
};

struct __align__(16) seqType16
{
	seqType4 a;
	seqType4 b;
	seqType4 c;
	seqType4 d;
};

struct __align__(16) scoreType8
{
	scoreType4 a;
	scoreType4 b;
};

struct __align__(32) scoreType16
{
	scoreType4 a;
	scoreType4 b;
	scoreType4 c;
	scoreType4 d;
};


struct __align__(4) TempData
{
	scoreType F;
	scoreType Ix;
};

struct __align__(8) TempData2
{
	TempData a;
	TempData b;
};

struct __align__(16) TempData4
{
	TempData a;
	TempData b;
	TempData c;
	TempData d;
};

bool launchSW(scoreType** scores, void* query, size_t queryLength, GPUdb& db, FastaMatrix& substitutionMatrix, int gapPenalty, int gapExtendPenalty, double& time);
