#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

typedef struct lyTensor
{
	char*		 name;
	int32_t*	 shape;
	int32_t		 rank;
	nv_bfloat16* data;
	size_t		 dataSize;
} lyTensor;

void lyTensorCreate(lyTensor** ppTensor, const int32_t* pShape, int32_t rank, const nv_bfloat16* pData, const char* name);
void lyTensorDestroy(lyTensor* pTensor);

void lyTensorReshape(lyTensor* pTensor, const int32_t* pShape, int32_t rank);
void lyTensorSlice(lyTensor** ppOutput, const lyTensor* pInput, int32_t startIdx, int32_t endIdx);

void lyTensorGetItem(float* pValue, const lyTensor* pTensor, const int32_t* pLoc);
void lyTensorSetItem(lyTensor* pTensor, const int32_t* pLoc, float value);
void lyTensorGetItemRaw(float* pValue, const lyTensor* pTensor, int32_t index);	 // Linear indices
void lyTensorSetItemRaw(lyTensor* pTensor, int32_t index, float value);

void lyTensorPrint(const lyTensor* pTensor);

typedef struct lyTensorFloat
{
	char*	 name;
	int32_t* shape;
	int32_t	 rank;
	float*	 data;
	size_t	 dataSize;
} lyTensorFloat;

void lyTensorFloatCreate(lyTensorFloat** ppTensor, const int32_t* pShape, int32_t rank, const float* pData, const char* name);
void lyTensorFloatDestroy(lyTensorFloat* pTensor);
void lyTensorFloatSlice(lyTensorFloat** ppOutput, const lyTensorFloat* pInput, int32_t startIdx, int32_t endIdx);

typedef struct lyTensorDouble
{
	char*	 name;
	int32_t* shape;
	int32_t	 rank;
	double*	 data;
	size_t	 dataSize;
} lyTensorDouble;

void lyTensorDoubleCreate(lyTensorDouble** ppTensor, const int32_t* pShape, int32_t rank, const double* pData, const char* name);
void lyTensorDoubleDestroy(lyTensorDouble* pTensor);
void lyTensorDoubleSlice(lyTensorDouble** ppOutput, const lyTensorDouble* pInput, int32_t startIdx, int32_t endIdx);
void lyTensorDoublePrint(lyTensorDouble* pTensor);