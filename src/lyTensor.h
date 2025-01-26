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

void lyTensorGetItem(int32_t* pValue, const lyTensor* pTensor, const int32_t* pLoc);
void lyTensorSetItem(lyTensor* pTensor, const int32_t* pLoc, int32_t value);
void lyTensorGetItemAsFloat32(float* pOut, const lyTensor* pTensor, int32_t index);
void lyTensorSetItemFromFloat32(lyTensor* pTensor, int32_t index, float value);
void lyTensorGetComplexItem(float* pReal, float* pImag, const lyTensor* pTensor, int32_t row, int32_t col);
void lyTensorSetComplexItem(lyTensor* pTensor, int32_t row, int32_t col, float real, float imag);

void lyTensorPrint(const lyTensor* pTensor);