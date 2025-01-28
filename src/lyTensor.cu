#include "lyTensor.h"

#include <float.h>
#include <math.h>
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

void lyTensorSetItem(lyTensor* pTensor, const int32_t* pLoc, float value)
{
	int32_t index		 = calculateIndex(pTensor, pLoc);
	pTensor->data[index] = __float2bfloat16_rz(value);
}

void lyTensorGetItem(float* pValue, const lyTensor* pTensor, const int32_t* pLoc)
{
	int32_t index = calculateIndex(pTensor, pLoc);
	*pValue		  = __bfloat162float(pTensor->data[index]);
}

void lyTensorSetItemRaw(lyTensor* pTensor, int32_t index, float value)
{
	pTensor->data[index] = __float2bfloat16_rz(value);
}

void lyTensorGetItemRaw(float* pValue, const lyTensor* pTensor, int32_t index)
{
	*pValue = __bfloat162float(pTensor->data[index]);
}

void lyTensorPrint(const lyTensor* pTensor)
{
	return;
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

	size_t totalElements = 1;
	for (int32_t i = 0; i < pTensor->rank; i++)
	{
		totalElements *= pTensor->shape[i];
	}

	if (!pTensor->data || totalElements == 0)
	{
		printf("[lyPrintTensor] No data to print.\n");
		return;
	}

	// Track finite stats separately
	float  sum		   = 0.0f;
	float  min		   = FLT_MAX;
	float  max		   = -FLT_MAX;
	size_t finiteCount = 0;
	size_t posInfCount = 0;
	size_t negInfCount = 0;

	for (size_t i = 0; i < totalElements; i++)
	{
		float val = __bfloat162float(pTensor->data[i]);
		if (isinf(val))
		{
			if (val > 0)
				posInfCount++;
			else
				negInfCount++;
		}
		else
		{
			sum += val;
			min = fminf(min, val);
			max = fmaxf(max, val);
			finiteCount++;
		}
	}

	printf("Stats:\n");
	if (finiteCount > 0)
	{
		float mean = sum / (float)finiteCount;

		// Calculate variance only for finite values
		float variance = 0.0f;
		for (size_t i = 0; i < totalElements; i++)
		{
			float val = __bfloat162float(pTensor->data[i]);
			if (!isinf(val))
			{
				float diff = val - mean;
				variance += diff * diff;
			}
		}
		variance /= (float)finiteCount;
		float std = sqrtf(variance);

		printf("  Finite values: %zu\n", finiteCount);
		printf("  Sum (finite): %.12f\n", sum);
		printf("  Mean (finite): %.12f\n", mean);
		printf("  Std (finite): %.12f\n", std);
		printf("  Min (finite): %.12f\n", min);
		printf("  Max (finite): %.12f\n", max);
	}
	else
	{
		printf("  No finite values\n");
	}

	if (posInfCount > 0 || negInfCount > 0)
	{
		printf("  Inf values: %zu (+), %zu (-)\n", posInfCount, negInfCount);
	}

	printf("First 5 elements: [");
	for (size_t i = 0; i < fmin(5, totalElements); i++)
	{
		float val = __bfloat162float(pTensor->data[i]);
		if (isinf(val))
		{
			if (val > 0)
				printf("Inf");
			else
				printf("-Inf");
		}
		else
		{
			printf("%.3f", val);
		}
		if (i < fmin(4, totalElements - 1))
			printf(", ");
	}
	printf("]\n");

	printf("Last 5 elements: [");
	for (size_t i = fmax(0, totalElements - 5); i < totalElements; i++)
	{
		float val = __bfloat162float(pTensor->data[i]);
		if (isinf(val))
		{
			if (val > 0)
				printf("Inf");
			else
				printf("-Inf");
		}
		else
		{
			printf("%.3f", val);
		}
		if (i < totalElements - 1)
			printf(", ");
	}
	printf("]\n");

	fflush(stdout);
}

void lyTensorFloatCreate(lyTensorFloat** ppTensor, const int32_t* pShape, int32_t rank, const float* pData, const char* name)
{
	lyTensorFloat* pTensor = (lyTensorFloat*)malloc(sizeof(lyTensorFloat));
	memset(pTensor, 0, sizeof(lyTensorFloat));

	if (pShape && rank > 0)
	{
		int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * rank);
		memcpy(newShape, pShape, sizeof(int32_t) * rank);
		pTensor->shape = newShape;
		pTensor->rank  = rank;

		size_t dataSize = 1;
		for (int32_t i = 0; i < rank; i++)
		{
			dataSize *= pShape[i];
		}
		dataSize *= sizeof(float);

		float* newData = (float*)malloc(dataSize);
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

void lyTensorFloatDestroy(lyTensorFloat* pTensor)
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

void lyTensorFloatPrint(const lyTensorFloat* pTensor)
{
	return;
	if (!pTensor)
	{
		printf("[lyTensorFloatPrint] Tensor is NULL.\n");
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

	float  sum			 = 0.0f;
	float  min			 = FLT_MAX;
	float  max			 = -FLT_MAX;
	float  mean			 = 0.0f;
	size_t totalElements = pTensor->dataSize / sizeof(float);

	for (size_t i = 0; i < totalElements; i++)
	{
		float val = pTensor->data[i];
		sum += val;
		min = fminf(min, val);
		max = fmaxf(max, val);
	}
	mean = sum / totalElements;

	float variance = 0.0f;
	for (size_t i = 0; i < totalElements; i++)
	{
		float diff = pTensor->data[i] - mean;
		variance += diff * diff;
	}
	variance /= totalElements;
	float std = sqrtf(variance);

	printf("Stats:\n");
	printf("  Sum: %.6f\n", sum);
	printf("  Mean: %.6f\n", mean);
	printf("  Std: %.6f\n", std);
	printf("  Min: %.6f\n", min);
	printf("  Max: %.6f\n", max);

	printf("First 5 elements: [");
	for (size_t i = 0; i < fmin(5, totalElements); i++)
	{
		printf("%.3f", pTensor->data[i]);
		if (i < 4)
			printf(", ");
	}
	printf("]\n");

	printf("Last 5 elements: [");
	for (size_t i = fmax(0, totalElements - 5); i < totalElements; i++)
	{
		printf("%.3f", pTensor->data[i]);
		if (i < totalElements - 1)
			printf(", ");
	}
	printf("]\n");

	fflush(stdout);
}

void lyTensorFloatSlice(lyTensorFloat** ppOutput, const lyTensorFloat* pInput, int32_t startIdx, int32_t endIdx)
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

	lyTensorFloat* pOutput;
	lyTensorFloatCreate(&pOutput, newShape, pInput->rank, pInput->data + offset, NULL);
	free(newShape);

	*ppOutput = pOutput;
}

void lyTensorDoubleCreate(lyTensorDouble** ppTensor, const int32_t* pShape, int32_t rank, const double* pData, const char* name)
{
	lyTensorDouble* pTensor = (lyTensorDouble*)malloc(sizeof(lyTensorDouble));
	memset(pTensor, 0, sizeof(lyTensorDouble));

	if (pShape && rank > 0)
	{
		int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * rank);
		memcpy(newShape, pShape, sizeof(int32_t) * rank);
		pTensor->shape = newShape;
		pTensor->rank  = rank;

		size_t dataSize = 1;
		for (int32_t i = 0; i < rank; i++)
		{
			dataSize *= pShape[i];
		}
		dataSize *= sizeof(double);

		double* newData = (double*)malloc(dataSize);
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

void lyTensorDoubleDestroy(lyTensorDouble* pTensor)
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

void lyTensorDoubleSlice(lyTensorDouble** ppOutput, const lyTensorDouble* pInput, int32_t startIdx, int32_t endIdx)
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

	lyTensorDouble* pOutput;
	lyTensorDoubleCreate(&pOutput, newShape, pInput->rank, pInput->data + offset, NULL);
	free(newShape);

	*ppOutput = pOutput;
}

void lyTensorDoublePrint(lyTensorDouble* pTensor)
{
	return;
	if (!pTensor)
	{
		printf("[lyTensorDoublePrint] Tensor is NULL.\n");
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

	// Calculate total elements from shape
	size_t totalElements = 1;
	for (int32_t i = 0; i < pTensor->rank; i++)
	{
		totalElements *= pTensor->shape[i];
	}

	if (!pTensor->data || totalElements == 0)
	{
		printf("[lyTensorDoublePrint] No data to print.\n");
		return;
	}

	// First pass - calculate sum, min, max
	double sum = 0.0;
	double min = DBL_MAX;
	double max = -DBL_MAX;

	for (size_t i = 0; i < totalElements; i++)
	{
		double val = pTensor->data[i];
		sum += val;
		min = fmin(min, val);
		max = fmax(max, val);
	}
	double mean = sum / (double)totalElements;

	// Second pass - calculate variance/std
	double variance = 0.0;
	for (size_t i = 0; i < totalElements; i++)
	{
		double val	= pTensor->data[i];
		double diff = val - mean;
		variance += diff * diff;
	}
	variance /= (double)totalElements;
	double std = sqrt(variance);

	printf("Stats:\n");
	printf("  Sum: %.6f\n", sum);
	printf("  Mean: %.6f\n", mean);
	printf("  Std: %.6f\n", std);
	printf("  Min: %.6f\n", min);
	printf("  Max: %.6f\n", max);

	printf("First 5 elements: [");
	for (size_t i = 0; i < fmin(5, totalElements); i++)
	{
		printf("%.3f", pTensor->data[i]);
		if (i < fmin(4, totalElements - 1))
			printf(", ");
	}
	printf("]\n");

	printf("Last 5 elements: [");
	for (size_t i = fmax(0, totalElements - 5); i < totalElements; i++)
	{
		printf("%.3f", pTensor->data[i]);
		if (i < totalElements - 1)
			printf(", ");
	}
	printf("]\n");

	fflush(stdout);
}