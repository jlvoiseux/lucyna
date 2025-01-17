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

bool lyTensorToGPU(lyTensor* pTensor);
bool lyTensorToCPU(lyTensor* pTensor);