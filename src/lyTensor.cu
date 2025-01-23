#include "lyTensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static bool validateIndex(lyTensor* pTensor, int32_t index, size_t elementSize)
{
	if (!pTensor || !pTensor->data)
	{
		return false;
	}

	size_t offset = index * elementSize;
	return offset < pTensor->dataSize;
}

static bool validateComplexAccess(lyTensor* pTensor, int32_t row, int32_t col)
{
	if (!pTensor || !pTensor->data || pTensor->rank != 2)
	{
		return false;
	}

	return row < pTensor->shape[0] && col < pTensor->shape[1];
}

static int32_t calculateIndex(lyTensor* pTensor, const int32_t* pLoc)
{
	int32_t index  = 0;
	int32_t stride = 1;

	for (int32_t i = pTensor->rank - 1; i >= 0; i--)
	{
		if (pLoc[i] >= pTensor->shape[i])
		{
			return -1;
		}
		index += pLoc[i] * stride;
		stride *= pTensor->shape[i];
	}

	return index;
}

__global__ void setItemFromInt32Kernel(nv_bfloat16* data, int32_t index, int32_t value)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		data[index] = __float2bfloat16((float)value);
	}
}

__global__ void setItemFromFloat32Kernel(nv_bfloat16* data, int32_t index, float value)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		data[index] = __float2bfloat16(value);
	}
}

__global__ void setComplexItemKernel(nv_bfloat16* data, int32_t baseIdx, float real, float imag)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		data[baseIdx]	  = __float2bfloat16(real);
		data[baseIdx + 1] = __float2bfloat16(imag);
	}
}

__global__ void getItemKernel(nv_bfloat16* result, const nv_bfloat16* data, int32_t index)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		result[0] = data[index];
	}
}

__global__ void getComplexItemKernel(nv_bfloat16* result, const nv_bfloat16* data, int32_t baseIdx)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		result[0] = data[baseIdx];
		result[1] = data[baseIdx + 1];
	}
}

bool lyCreateTensor(lyTensor** ppTensor, lyMemoryType memoryType)
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
	pTensor->memoryType = memoryType;

	*ppTensor = pTensor;
	return true;
}

void lyFreeTensorData(lyTensor* pTensor)
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
	}
	pTensor->data = NULL;
}

void lyDestroyTensor(lyTensor* pTensor)
{
	lyFreeTensorData(pTensor);
	free(pTensor);
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

static bool checkGpuMemory(const char* operation, size_t requestedBytes)
{
	size_t		free, total;
	cudaError_t error = cudaMemGetInfo(&free, &total);
	if (error != cudaSuccess)
	{
		printf("Failed to query GPU memory: %s\n", cudaGetErrorString(error));
		return false;
	}

	double freeGb	   = free / (1024.0 * 1024.0 * 1024.0);
	double totalGb	   = total / (1024.0 * 1024.0 * 1024.0);
	double requestedGb = requestedBytes / (1024.0 * 1024.0 * 1024.0);

	printf("GPU Memory Status for %s:\n", operation);
	printf("  Total: %.2f GB\n", totalGb);
	printf("  Free: %.2f GB\n", freeGb);
	printf("  Requested: %.2f GB\n", requestedGb);

	return free >= requestedBytes;
}

bool lySetTensorData(lyTensor* pTensor, const nv_bfloat16* pData, size_t dataSize)
{
	if (!pTensor)
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

	if (pTensor->memoryType == LY_MEMORY_GPU)
	{
		//		if (!checkGpuMemory("tensor allocation", dataSize))
		//		{
		//			printf("Not enough GPU memory for tensor allocation\n");
		//			return false;
		//		}

		nv_bfloat16* gpuData	 = NULL;
		size_t		 alignedSize = (dataSize + 15) & ~15;  // Align to 16 bytes
		cudaError_t	 error		 = cudaMalloc(&gpuData, alignedSize);
		if (error != cudaSuccess)
		{
			printf("CUDA malloc failed: %s\n", cudaGetErrorString(error));
			return false;
		}

		if (pData)
		{
			error = cudaMemcpy(gpuData, pData, dataSize, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
			{
				printf("CUDA memcpy failed: %s\n", cudaGetErrorString(error));
				cudaFree(gpuData);
				return false;
			}
		}
		else
		{
			error = cudaMemset(gpuData, 0, dataSize);
			if (error != cudaSuccess)
			{
				printf("CUDA memset failed: %s\n", cudaGetErrorString(error));
				cudaFree(gpuData);
				return false;
			}
		}

		pTensor->data = gpuData;
	}
	else
	{
		nv_bfloat16* cpuData = (nv_bfloat16*)malloc(dataSize);
		if (!cpuData)
		{
			return false;
		}

		if (pData)
		{
			memcpy(cpuData, pData, dataSize);
		}
		else
		{
			memset(cpuData, 0, dataSize);
		}

		pTensor->data = cpuData;
	}

	pTensor->dataSize = dataSize;
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

bool lyTensorSlice(lyTensor** ppOutput, lyTensor* pInput, int32_t startIdx, int32_t endIdx)
{
	if (!ppOutput || !pInput || startIdx < 0 || endIdx <= startIdx || endIdx > pInput->shape[0] || !pInput->data)
	{
		return false;
	}

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput, LY_MEMORY_CPU))
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
	size_t offset	 = startIdx * sliceElements;

	if (!lySetTensorData(pOutput, pInput->data + offset, (endIdx - startIdx) * sliceSize))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorSetItem(lyTensor* pTensor, const int32_t* pLoc, int32_t value)
{
	if (pTensor->memoryType == LY_MEMORY_CPU)
	{
		lyTensorMoveToGPU(pTensor);
	}

	if (!pTensor || !pLoc)
	{
		return false;
	}

	int32_t index = calculateIndex(pTensor, pLoc);
	if (index < 0)
	{
		return false;
	}

	cudaDeviceSynchronize();
	setItemFromInt32Kernel<<<1, 1>>>(pTensor->data, index, value);

	lyTensorMoveToCPU(pTensor);

	return cudaGetLastError() == cudaSuccess;
}

bool lyTensorGetItem(int32_t* pValue, lyTensor* pTensor, const int32_t* pLoc)
{
	if (!pValue || !pTensor || !pLoc)
	{
		return false;
	}

	int32_t index = calculateIndex(pTensor, pLoc);
	if (index < 0)
	{
		return false;
	}

	float value;
	if (!lyTensorGetItemAsFloat32(&value, pTensor, index))
	{
		return false;
	}

	*pValue = (int32_t)value;
	return true;
}

bool lyTensorGetItemAsFloat32(float* pOut, lyTensor* pTensor, int32_t index)
{
	if (pTensor->memoryType == LY_MEMORY_CPU)
	{
		lyTensorMoveToGPU(pTensor);
	}

	if (!pOut || !validateIndex(pTensor, index, sizeof(nv_bfloat16)))
	{
		return false;
	}

	nv_bfloat16* temp;
	if (cudaMalloc(&temp, sizeof(nv_bfloat16)) != cudaSuccess)
	{
		return false;
	}

	cudaDeviceSynchronize();
	getItemKernel<<<1, 1>>>(temp, pTensor->data, index);

	nv_bfloat16 hostValue;
	if (cudaMemcpy(&hostValue, temp, sizeof(nv_bfloat16), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cudaFree(temp);
		return false;
	}

	cudaFree(temp);
	*pOut = __bfloat162float(hostValue);

	lyTensorMoveToCPU(pTensor);

	return true;
}

bool lyTensorSetItemFromFloat32(lyTensor* pTensor, int32_t index, float value)
{
	if (pTensor->memoryType == LY_MEMORY_CPU)
	{
		lyTensorMoveToGPU(pTensor);
	}

	if (!validateIndex(pTensor, index, sizeof(nv_bfloat16)))
	{
		return false;
	}

	cudaDeviceSynchronize();
	setItemFromFloat32Kernel<<<1, 1>>>(pTensor->data, index, value);

	lyTensorMoveToCPU(pTensor);

	return cudaGetLastError() == cudaSuccess;
}

bool lyTensorGetComplexItem(float* pReal, float* pImag, lyTensor* pTensor, int32_t row, int32_t col)
{
	if (pTensor->memoryType == LY_MEMORY_CPU)
	{
		lyTensorMoveToGPU(pTensor);
	}

	if (!pReal || !pImag || !validateComplexAccess(pTensor, row, col))
	{
		return false;
	}

	int32_t baseIdx = row * pTensor->shape[1] * 2 + col * 2;

	nv_bfloat16* temp;
	if (cudaMalloc(&temp, 2 * sizeof(nv_bfloat16)) != cudaSuccess)
	{
		return false;
	}

	cudaDeviceSynchronize();
	getComplexItemKernel<<<1, 1>>>(temp, pTensor->data, baseIdx);

	nv_bfloat16 hostValues[2];
	if (cudaMemcpy(hostValues, temp, 2 * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cudaFree(temp);
		return false;
	}

	cudaFree(temp);
	*pReal = __bfloat162float(hostValues[0]);
	*pImag = __bfloat162float(hostValues[1]);

	lyTensorMoveToCPU(pTensor);

	return true;
}

bool lyTensorSetComplexItem(lyTensor* pTensor, int32_t row, int32_t col, float real, float imag)
{
	if (pTensor->memoryType == LY_MEMORY_CPU)
	{
		lyTensorMoveToGPU(pTensor);
	}

	if (!validateComplexAccess(pTensor, row, col))
	{
		return false;
	}

	int32_t baseIdx = row * pTensor->shape[1] * 2 + col * 2;

	cudaDeviceSynchronize();
	setComplexItemKernel<<<1, 1>>>(pTensor->data, baseIdx, real, imag);

	lyTensorMoveToCPU(pTensor);

	return cudaGetLastError() == cudaSuccess;
}

void lyTensorPrint(lyTensor* pTensor)
{
	if (!pTensor || !pTensor->data)
	{
		printf("Tensor is null or uninitialized.\n");
		return;
	}

	printf("Tensor Metadata:\n");
	printf("Name: %s\n", pTensor->name ? pTensor->name : "Unnamed");
	printf("Rank: %d\n", pTensor->rank);
	printf("Shape: ");
	for (int32_t i = 0; i < pTensor->rank; i++)
	{
		printf("%d%s", pTensor->shape[i], (i < pTensor->rank - 1) ? " x " : "\n");
	}

	if (pTensor->rank > 3)
	{
		printf("Error: Printing tensors with rank > 3 is not supported.\n");
		return;
	}

	size_t elements = 1;
	for (int32_t i = 0; i < pTensor->rank; i++)
	{
		elements *= pTensor->shape[i];
	}

	nv_bfloat16* hostData = (nv_bfloat16*)malloc(pTensor->dataSize);
	if (!hostData)
	{
		printf("Error: Failed to allocate memory for tensor data.\n");
		return;
	}

	if (cudaMemcpy(hostData, pTensor->data, pTensor->dataSize, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("Error: Failed to copy tensor data from device to host.\n");
		free(hostData);
		return;
	}

	printf("Tensor Data:\n");
	if (pTensor->rank == 1)
	{
		for (int32_t i = 0; i < pTensor->shape[0]; i++)
		{
			printf("%f ", __bfloat162float(hostData[i]));
		}
		printf("\n");
	}
	else if (pTensor->rank == 2)
	{
		for (int32_t i = 0; i < pTensor->shape[0]; i++)
		{
			for (int32_t j = 0; j < pTensor->shape[1]; j++)
			{
				printf("%f ", __bfloat162float(hostData[i * pTensor->shape[1] + j]));
			}
			printf("\n");
		}
	}
	else if (pTensor->rank == 3)
	{
		for (int32_t i = 0; i < pTensor->shape[0]; i++)
		{
			printf("Slice %d:\n", i);
			for (int32_t j = 0; j < pTensor->shape[1]; j++)
			{
				for (int32_t k = 0; k < pTensor->shape[2]; k++)
				{
					printf("%f ", __bfloat162float(hostData[i * pTensor->shape[1] * pTensor->shape[2] + j * pTensor->shape[2] + k]));
				}
				printf("\n");
			}
			printf("\n");
		}
	}

	free(hostData);
}

bool lyTensorMoveToGPU(lyTensor* pTensor)
{
	cudaDeviceSynchronize();
	if (!pTensor || pTensor->memoryType != LY_MEMORY_CPU || !pTensor->data)
	{
		return false;
	}

	//	if (!checkGpuMemory("tensor allocation", pTensor->dataSize))
	//	{
	//		printf("Not enough GPU memory for tensor allocation\n");
	//		return false;
	//	}

	nv_bfloat16* gpuData = NULL;
	cudaError_t	 error	 = cudaMalloc(&gpuData, pTensor->dataSize);
	if (error != cudaSuccess)
	{
		return false;
	}

	error = cudaMemcpy(gpuData, pTensor->data, pTensor->dataSize, cudaMemcpyHostToDevice);
	if (error != cudaSuccess)
	{
		cudaFree(gpuData);
		return false;
	}

	free(pTensor->data);
	pTensor->data		= gpuData;
	pTensor->memoryType = LY_MEMORY_GPU;
	cudaDeviceSynchronize();
	return true;
}

bool lyTensorMoveToCPU(lyTensor* pTensor)
{
	cudaDeviceSynchronize();
	if (!pTensor || pTensor->memoryType != LY_MEMORY_GPU || !pTensor->data)
	{
		return false;
	}

	nv_bfloat16* cpuData = (nv_bfloat16*)malloc(pTensor->dataSize);
	if (!cpuData)
	{
		return false;
	}

	cudaError_t error = cudaMemcpy(cpuData, pTensor->data, pTensor->dataSize, cudaMemcpyDeviceToHost);
	if (error != cudaSuccess)
	{
		free(cpuData);
		return false;
	}

	cudaFree(pTensor->data);
	pTensor->data		= cpuData;
	pTensor->memoryType = LY_MEMORY_CPU;

	cudaDeviceSynchronize();
	return true;
}