#include "lyTensor.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t calculateIndex(const lyTensor* pTensor, const int32_t* pLoc)
{
	int32_t index  = 0;
	int32_t stride = 1;

	for (int32_t i = pTensor->rank - 1; i >= 0; i--)
	{
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

void lyTensorCreate(lyTensor** ppTensor, const int32_t* pShape, int32_t rank, const nv_bfloat16* pData, const char* name)
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

void lyTensorDestroy(lyTensor* pTensor)
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

void lyTensorReshape(lyTensor* pTensor, const int32_t* pShape, int32_t rank)
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

void lyTensorSlice(lyTensor** ppOutput, const lyTensor* pInput, int32_t startIdx, int32_t endIdx)
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
	lyTensorCreate(&pOutput, newShape, pInput->rank, pInput->data + offset, NULL);
	free(newShape);

	*ppOutput = pOutput;
}

void lyTensorSetItem(lyTensor* pTensor, const int32_t* pLoc, int32_t value)
{
	int32_t index		 = calculateIndex(pTensor, pLoc);
	pTensor->data[index] = __float2bfloat16((float)value);
}

void lyTensorGetItem(int32_t* pValue, const lyTensor* pTensor, const int32_t* pLoc)
{
	int32_t index = calculateIndex(pTensor, pLoc);
	float	value;
	lyTensorGetItemAsFloat32(&value, pTensor, index);
	*pValue = (int32_t)value;
}

void lyTensorGetItemAsFloat32(float* pOut, const lyTensor* pTensor, int32_t index)
{
	*pOut = __bfloat162float(pTensor->data[index]);
}

void lyTensorSetItemFromFloat32(lyTensor* pTensor, int32_t index, float value)
{
	pTensor->data[index] = __float2bfloat16(value);
}

void lyTensorGetComplexItem(float* pReal, float* pImag, const lyTensor* pTensor, int32_t row, int32_t col)
{
	int32_t baseIdx = row * pTensor->shape[1] + col * 2;
	*pReal			= __bfloat162float(pTensor->data[baseIdx]);
	*pImag			= __bfloat162float(pTensor->data[baseIdx + 1]);
}

void lyTensorSetComplexItem(lyTensor* pTensor, int32_t row, int32_t col, float real, float imag)
{
	int32_t baseIdx			   = row * pTensor->shape[1] + col * 2;
	pTensor->data[baseIdx]	   = __float2bfloat16(real);
	pTensor->data[baseIdx + 1] = __float2bfloat16(imag);
}

void lyTensorPrint(const lyTensor* pTensor)
{
	if (!pTensor)
	{
		printf("[lyPrintTensor] Tensor is NULL.\n");
		return;
	}

	if (pTensor->name)
	{
		printf("Tensor Name: %s\n", pTensor->name);
	}
	else
	{
		printf("Tensor Name: [Unnamed]\n");
	}

	printf("Tensor Rank: %d\n", pTensor->rank);
	printf("Tensor Shape: [");
	for (int32_t i = 0; i < pTensor->rank; i++)
	{
		printf("%d", pTensor->shape[i]);
		if (i < pTensor->rank - 1)
		{
			printf(", ");
		}
	}
	printf("]\n");
	printf("Tensor Data (first few elements):\n");

	size_t limit = 10;
	size_t count = 0;

	printf("[");
	for (size_t i = 0; i < pTensor->dataSize / sizeof(nv_bfloat16); i++)
	{
		float value = __bfloat162float(pTensor->data[i]);
		printf("%.3f", value);
		count++;

		if (count >= limit || i == (pTensor->dataSize / sizeof(nv_bfloat16)) - 1)
		{
			break;
		}
		else
		{
			printf(", ");
		}
	}
	printf("]\n");

	fflush(stdout);
}