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

static size_t calculateSize(const int32_t* pShape, int32_t rank)
{
	size_t size = 1;
	for (int32_t i = 0; i < rank; i++)
	{
		size *= pShape[i];
	}
	return size * sizeof(nv_bfloat16);
}

void lyCreateTensor(lyTensor** ppTensor, const int32_t* pShape, int32_t rank, const nv_bfloat16* pData, const char* name)
{
	lyTensor* pTensor = (lyTensor*)malloc(sizeof(lyTensor));
	memset(pTensor, 0, sizeof(lyTensor));

	if (pShape && rank > 0)
	{
		int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * rank);
		memcpy(newShape, pShape, sizeof(int32_t) * rank);
		pTensor->shape = newShape;
		pTensor->rank  = rank;

		size_t		 dataSize = calculateSize(pShape, rank);
		nv_bfloat16* newData  = (nv_bfloat16*)malloc(dataSize);
		if (pData)
			memcpy(newData, pData, dataSize);
		else
			memset(newData, 0, dataSize);
		pTensor->data	  = newData;
		pTensor->dataSize = dataSize;
	}

	if (name)
		pTensor->name = strdup(name);

	*ppTensor = pTensor;
}

void lyDestroyTensor(lyTensor* pTensor)
{
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
		free(pTensor->data);
	}
	free(pTensor);
}

void lyReshapeTensor(lyTensor* pTensor, const int32_t* pShape, int32_t rank)
{
	int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * rank);
	int32_t	 oldSize  = 1;
	for (int32_t i = 0; i < pTensor->rank; i++)
		oldSize *= pTensor->shape[i];

	int32_t newSize = 1;
	for (int32_t i = 0; i < rank; i++)
	{
		newSize *= pShape[i];
		newShape[i] = pShape[i];
	}

	free(pTensor->shape);
	pTensor->shape = newShape;
	pTensor->rank  = rank;
}

void lyTensorSlice(lyTensor** ppOutput, lyTensor* pInput, int32_t startIdx, int32_t endIdx)
{
	size_t sliceElements = 1;
	for (int32_t i = 1; i < pInput->rank; i++)
	{
		sliceElements *= pInput->shape[i];
	}
	size_t offset = startIdx * sliceElements;

	int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * pInput->rank);
	memcpy(newShape, pInput->shape, sizeof(int32_t) * pInput->rank);
	newShape[0] = endIdx - startIdx;

	lyTensor* pOutput;
	lyCreateTensor(&pOutput, newShape, pInput->rank, pInput->data + offset, NULL);
	free(newShape);

	*ppOutput = pOutput;
}

bool lyTensorSetItem(lyTensor* pTensor, const int32_t* pLoc, int32_t value)
{
	if (!pTensor || !pLoc)
	{
		return false;
	}

	int32_t index = calculateIndex(pTensor, pLoc);
	if (index < 0)
	{
		return false;
	}

	pTensor->data[index] = __float2bfloat16((float)value);
	return true;
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
	if (!pOut || !validateIndex(pTensor, index, sizeof(nv_bfloat16)))
	{
		return false;
	}

	*pOut = __bfloat162float(pTensor->data[index]);
	return true;
}

bool lyTensorSetItemFromFloat32(lyTensor* pTensor, int32_t index, float value)
{
	if (!validateIndex(pTensor, index, sizeof(nv_bfloat16)))
	{
		return false;
	}

	pTensor->data[index] = __float2bfloat16(value);
	return true;
}

bool lyTensorGetComplexItem(float* pReal, float* pImag, lyTensor* pTensor, int32_t row, int32_t col)
{
	if (!pReal || !pImag || !validateComplexAccess(pTensor, row, col))
	{
		return false;
	}

	int32_t baseIdx = row * pTensor->shape[1] * 2 + col * 2;

	*pReal = __bfloat162float(pTensor->data[baseIdx]);
	*pImag = __bfloat162float(pTensor->data[baseIdx + 1]);

	return true;
}

bool lyTensorSetComplexItem(lyTensor* pTensor, int32_t row, int32_t col, float real, float imag)
{
	if (!validateComplexAccess(pTensor, row, col))
	{
		return false;
	}

	int32_t baseIdx = row * pTensor->shape[1] * 2 + col * 2;

	pTensor->data[baseIdx]	   = __float2bfloat16(real);
	pTensor->data[baseIdx + 1] = __float2bfloat16(imag);

	return true;
}