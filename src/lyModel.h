#pragma once

#include "lyTensor.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

typedef struct lyMappedFile
{
#ifdef _WIN32
	HANDLE fileHandle;
	HANDLE mappingHandle;
#else
	int fd;
#endif
	void*  mappedMemory;
	size_t mappedSize;
} lyMappedFile;

typedef struct lyModelArgs
{
	int32_t dim;
	int32_t nLayers;
	int32_t nHeads;
	int32_t nKVHeads;
	int32_t vocabSize;
	int32_t multipleOf;
	float	ffnDimMultiplier;
	float	normEps;
	bool	useScaledRope;
	float	ropeTheta;
	int32_t maxSequenceLength;
	int32_t nRep;
	int32_t headDim;
} lyModelArgs;

typedef struct lyModel
{
	lyModelArgs	 args;
	lyTensor*	 tensors;
	int32_t		 tensorCount;
	lyMappedFile mapping;
} lyModel;

bool lyCreateDefaultModelArgs(lyModelArgs* pArgs);
bool lyGetModelTensor(lyTensor** ppTensor, const lyModel* pModel, const char* name);