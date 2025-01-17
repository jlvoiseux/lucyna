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

	// Free existing data if any
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