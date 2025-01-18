#include "lyTensor.h"

#include <stdlib.h>
#include <string.h>

bool lyCreateTensor(lyTensor** ppTensor)
{
	if (!ppTensor)
	{
		return false;
	}

	lyTensor* pTensor = (lyTensor*)malloc(sizeof(lyTensor));
	if (!pTensor)
	{
		return false;
	}

	memset(pTensor, 0, sizeof(lyTensor));
	pTensor->name		= NULL;
	pTensor->shape		= NULL;
	pTensor->rank		= 0;
	pTensor->data		= NULL;
	pTensor->dataSize	= 0;
	pTensor->memoryType = LY_MEMORY_CPU;

	*ppTensor = pTensor;
	return true;
}

void lyDestroyTensor(lyTensor* pTensor)
{
	if (!pTensor)
	{
		return;
	}

	if (pTensor->name)
	{
		free(pTensor->name);
		pTensor->name = NULL;
	}

	if (pTensor->shape)
	{
		free(pTensor->shape);
		pTensor->shape = NULL;
	}

	if (pTensor->data)
	{
		if (pTensor->memoryType == LY_MEMORY_GPU)
		{
			cudaFree(pTensor->data);
		}
		else
		{
			free(pTensor->data);
		}
		pTensor->data = NULL;
	}
}

bool lySetTensorShape(lyTensor* pTensor, const int32_t* pShape, int32_t rank)
{
	if (!pTensor || !pShape || rank <= 0)
	{
		return false;
	}

	int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * rank);
	if (!newShape)
	{
		return false;
	}

	memcpy(newShape, pShape, sizeof(int32_t) * rank);

	free(pTensor->shape);
	pTensor->shape = newShape;
	pTensor->rank  = rank;

	return true;
}

bool lySetTensorData(lyTensor* pTensor, const nv_bfloat16* pData, size_t dataSize, lyMemoryType memoryType)
{
	if (!pTensor || !pData || dataSize == 0)
	{
		return false;
	}

	if (pTensor->data)
	{
		if (pTensor->memoryType == LY_MEMORY_GPU)
		{
			cudaFree(pTensor->data);
		}
		else
		{
			free(pTensor->data);
		}
		pTensor->data = NULL;
	}

	void* newData;
	if (memoryType == LY_MEMORY_CPU)
	{
		newData = malloc(dataSize);
		if (!newData)
		{
			return false;
		}
		memcpy(newData, pData, dataSize);
	}
	else
	{
		if (cudaMalloc(&newData, dataSize) != cudaSuccess)
		{
			return false;
		}
		if (cudaMemcpy(newData, pData, dataSize, cudaMemcpyDefault) != cudaSuccess)
		{
			cudaFree(newData);
			return false;
		}
	}

	pTensor->data		= (nv_bfloat16*)newData;
	pTensor->dataSize	= dataSize;
	pTensor->memoryType = memoryType;

	return true;
}

bool lySetTensorName(lyTensor* pTensor, const char* name)
{
	if (!pTensor || !name)
	{
		return false;
	}

	char* newName = (char*)malloc(strlen(name) + 1);
	if (!newName)
	{
		return false;
	}

	strcpy(newName, name);
	free(pTensor->name);
	pTensor->name = newName;

	return true;
}

bool lyReshapeTensor(lyTensor* pTensor, const int32_t* pShape, int32_t rank)
{
	if (!pTensor || !pShape || rank <= 0)
	{
		return false;
	}

	int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * rank);
	if (!newShape)
	{
		return false;
	}

	int32_t oldSize = 1;
	for (int32_t i = 0; i < pTensor->rank; i++)
	{
		oldSize *= pTensor->shape[i];
	}

	int32_t newSize = 1;
	for (int32_t i = 0; i < rank; i++)
	{
		newSize *= pShape[i];
		newShape[i] = pShape[i];
	}

	if (oldSize != newSize)
	{
		free(newShape);
		return false;
	}

	free(pTensor->shape);
	pTensor->shape = newShape;
	pTensor->rank  = rank;

	return true;
}

bool lyTensorSlice(lyTensor** ppOutput, const lyTensor* pInput, int32_t startIdx, int32_t endIdx)
{
	if (!ppOutput || !pInput || startIdx < 0 || endIdx <= startIdx || endIdx > pInput->shape[0] || !pInput->data)
	{
		return false;
	}

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * pInput->rank);
	if (!newShape)
	{
		lyDestroyTensor(pOutput);
		return false;
	}
	memcpy(newShape, pInput->shape, sizeof(int32_t) * pInput->rank);
	newShape[0] = endIdx - startIdx;

	if (!lySetTensorShape(pOutput, newShape, pInput->rank))
	{
		free(newShape);
		lyDestroyTensor(pOutput);
		return false;
	}
	free(newShape);

	size_t sliceElements = 1;
	for (int32_t i = 1; i < pInput->rank; i++)
	{
		sliceElements *= pInput->shape[i];
	}
	size_t sliceSize = sliceElements * sizeof(nv_bfloat16);
	size_t offset	 = startIdx * sliceSize;

	size_t outputSize = (endIdx - startIdx) * sliceSize;
	if (cudaMalloc(&pOutput->data, outputSize) != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	if (cudaMemcpy(pOutput->data, pInput->data + (offset / sizeof(nv_bfloat16)), outputSize, cudaMemcpyDeviceToDevice) != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	pOutput->memoryType = LY_MEMORY_GPU;
	*ppOutput			= pOutput;
	return true;
}

bool lyTensorGetItemAsFloat32(float* pOut, const lyTensor* pTensor, int32_t index)
{
	if (!pOut || !pTensor || !pTensor->data)
	{
		return false;
	}

	size_t offset = index * sizeof(nv_bfloat16);
	if (offset >= pTensor->dataSize)
	{
		return false;
	}

	*pOut = __bfloat162float(pTensor->data[index]);
	return true;
}

bool lyTensorSetItemFromFloat32(lyTensor* pTensor, int32_t index, float value)
{
	if (!pTensor || !pTensor->data)
	{
		return false;
	}

	size_t offset = index * sizeof(nv_bfloat16);
	if (offset >= pTensor->dataSize)
	{
		return false;
	}

	pTensor->data[index] = __float2bfloat16(value);
	return true;
}

bool lyTensorGetComplexItem(float* pReal, float* pImag, const lyTensor* pTensor, int32_t row, int32_t col)
{
	if (!pReal || !pImag || !pTensor || !pTensor->data || pTensor->rank != 2)
	{
		return false;
	}

	if (row >= pTensor->shape[0] || col >= pTensor->shape[1])
	{
		return false;
	}

	// Complex numbers are stored as consecutive real/imag pairs
	int32_t baseIdx = row * pTensor->shape[1] * 2 + col * 2;

	*pReal = __bfloat162float(pTensor->data[baseIdx]);
	*pImag = __bfloat162float(pTensor->data[baseIdx + 1]);

	return true;
}

bool lyTensorSetComplexItem(lyTensor* pTensor, int32_t row, int32_t col, float real, float imag)
{
	if (!pTensor || !pTensor->data || pTensor->rank != 2)
	{
		return false;
	}

	if (row >= pTensor->shape[0] || col >= pTensor->shape[1])
	{
		return false;
	}

	int32_t baseIdx = row * pTensor->shape[1] * 2 + col * 2;

	pTensor->data[baseIdx]	   = __float2bfloat16(real);
	pTensor->data[baseIdx + 1] = __float2bfloat16(imag);

	return true;
}

bool lyTensorToGPU(lyTensor* pTensor)
{
	if (!pTensor || !pTensor->data || pTensor->memoryType == LY_MEMORY_GPU)
	{
		return false;
	}

	void* gpuData;
	if (cudaMalloc(&gpuData, pTensor->dataSize) != cudaSuccess)
	{
		return false;
	}

	if (cudaMemcpy(gpuData, pTensor->data, pTensor->dataSize, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cudaFree(gpuData);
		return false;
	}

	free(pTensor->data);
	pTensor->data		= (nv_bfloat16*)gpuData;
	pTensor->memoryType = LY_MEMORY_GPU;

	return true;
}

bool lyTensorToCPU(lyTensor* pTensor)
{
	if (!pTensor || !pTensor->data || pTensor->memoryType == LY_MEMORY_CPU)
	{
		return false;
	}

	void* cpuData = malloc(pTensor->dataSize);
	if (!cpuData)
	{
		return false;
	}

	if (cudaMemcpy(cpuData, pTensor->data, pTensor->dataSize, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		free(cpuData);
		return false;
	}

	cudaFree(pTensor->data);
	pTensor->data		= (nv_bfloat16*)cpuData;
	pTensor->memoryType = LY_MEMORY_CPU;

	return true;
}