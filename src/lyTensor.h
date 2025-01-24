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

void lyCreateTensor(lyTensor** ppTensor, const int32_t* pShape, int32_t rank, const nv_bfloat16* pData, const char* name);
void lyDestroyTensor(lyTensor* pTensor);

void lyReshapeTensor(lyTensor* pTensor, const int32_t* pShape, int32_t rank);
void lyTensorSlice(lyTensor** ppOutput, lyTensor* pInput, int32_t startIdx, int32_t endIdx);

bool lyTensorGetItem(int32_t* pValue, lyTensor* pTensor, const int32_t* pLoc);
bool lyTensorSetItem(lyTensor* pTensor, const int32_t* pLoc, int32_t value);
bool lyTensorGetItemAsFloat32(float* pOut, lyTensor* pTensor, int32_t index);
bool lyTensorSetItemFromFloat32(lyTensor* pTensor, int32_t index, float value);
bool lyTensorGetComplexItem(float* pReal, float* pImag, lyTensor* pTensor, int32_t row, int32_t col);
bool lyTensorSetComplexItem(lyTensor* pTensor, int32_t row, int32_t col, float real, float imag);