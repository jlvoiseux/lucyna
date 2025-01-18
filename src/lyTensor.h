#pragma once

#include <cuda_bf16.h>
#include <stdint.h>

typedef enum lyMemoryType
{
	LY_MEMORY_CPU,
	LY_MEMORY_GPU
} lyMemoryType;

typedef struct lyTensor
{
	char*		 name;
	int32_t*	 shape;
	int32_t		 rank;
	nv_bfloat16* data;
	size_t		 dataSize;
	lyMemoryType memoryType;
} lyTensor;

bool lyCreateTensor(lyTensor** ppTensor);
void lyDestroyTensor(lyTensor* pTensor);

bool lySetTensorShape(lyTensor* pTensor, const int32_t* pShape, int32_t rank);
bool lySetTensorData(lyTensor* pTensor, const nv_bfloat16* pData, size_t dataSize, lyMemoryType memoryType);
bool lySetTensorName(lyTensor* pTensor, const char* name);

bool lyReshapeTensor(lyTensor* pTensor, const int32_t* pShape, int32_t rank);
bool lyTensorSlice(lyTensor** ppOutput, const lyTensor* pInput, int32_t startIdx, int32_t endIdx);

bool lyTensorGetItemAsFloat32(float* pOut, const lyTensor* pTensor, int32_t index);
bool lyTensorSetItemFromFloat32(lyTensor* pTensor, int32_t index, float value);
bool lyTensorGetComplexItem(float* pReal, float* pImag, const lyTensor* pTensor, int32_t row, int32_t col);
bool lyTensorSetComplexItem(lyTensor* pTensor, int32_t row, int32_t col, float real, float imag);

bool lyTensorToGPU(lyTensor* pTensor);
bool lyTensorToCPU(lyTensor* pTensor);