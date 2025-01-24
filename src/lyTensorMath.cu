#include "lyTensorMath.h"

#include <cuda_bf16.h>
#include <math_constants.h>
#include <stdio.h>

bool lyTensorMatMul(lyTensor** ppOutput, lyTensor* pA, lyTensor* pB)
{
	if (!ppOutput || !pA || !pB || pA->rank < 2 || pB->rank < 2)
		return false;

	if (pA->rank != pB->rank)
		return false;

	for (int32_t i = 0; i < pA->rank - 2; i++)
	{
		if (pA->shape[i] != pB->shape[i])
			return false;
	}

	int32_t m = pA->shape[pA->rank - 2];
	int32_t k = pA->shape[pA->rank - 1];
	int32_t n = pB->shape[pB->rank - 1];

	if (k != pB->shape[pB->rank - 2])
		return false;

	int32_t batchSize = 1;
	for (int32_t i = 0; i < pA->rank - 2; i++)
	{
		batchSize *= pA->shape[i];
	}

	int32_t* outShape = (int32_t*)malloc(sizeof(int32_t) * pA->rank);
	if (!outShape)
		return false;

	for (int32_t i = 0; i < pA->rank - 2; i++)
	{
		outShape[i] = pA->shape[i];
	}
	outShape[pA->rank - 2] = m;
	outShape[pA->rank - 1] = n;

	lyTensor* pOutput;
	lyCreateTensor(&pOutput, outShape, pA->rank, NULL, NULL);
	free(outShape);

	for (int32_t batch = 0; batch < batchSize; batch++)
	{
		nv_bfloat16* pBatchA   = pA->data + batch * m * k;
		nv_bfloat16* pBatchB   = pB->data + batch * k * n;
		nv_bfloat16* pBatchOut = pOutput->data + batch * m * n;

		for (int32_t i = 0; i < m; i++)
		{
			for (int32_t j = 0; j < n; j++)
			{
				float sum = 0.0f;
				for (int32_t l = 0; l < k; l++)
				{
					float aVal = __bfloat162float(pBatchA[i * k + l]);
					float bVal = __bfloat162float(pBatchB[l * n + j]);
					sum += aVal * bVal;
				}
				pBatchOut[i * n + j] = __float2bfloat16(sum);
			}
		}
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorScaleAndAdd(lyTensor** ppOutput, lyTensor* pA, lyTensor* pB, float alpha, float beta)
{
	if (!ppOutput || !pA || !pB || !pA->data || !pB->data || pA->rank < 2 || pB->rank < 2)
		return false;

	if (pB->rank > pA->rank)
		return false;

	for (int32_t i = 0; i < pB->rank; i++)
	{
		int32_t aIdx = pA->rank - pB->rank + i;
		if (pB->shape[i] != pA->shape[aIdx])
			return false;
	}

	lyTensor* pOutput;
	lyCreateTensor(&pOutput, pA->shape, pA->rank, NULL, NULL);

	int32_t totalElements = 1;
	for (int32_t i = 0; i < pA->rank; i++)
	{
		totalElements *= pA->shape[i];
	}

	int32_t* bStrides = (int32_t*)malloc(pB->rank * sizeof(int32_t));
	if (!bStrides)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	bStrides[pB->rank - 1] = 1;
	for (int32_t i = pB->rank - 2; i >= 0; i--)
	{
		bStrides[i] = bStrides[i + 1] * pB->shape[i + 1];
	}

	int32_t* aStrides = (int32_t*)malloc(pA->rank * sizeof(int32_t));
	if (!aStrides)
	{
		free(bStrides);
		lyDestroyTensor(pOutput);
		return false;
	}

	aStrides[pA->rank - 1] = 1;
	for (int32_t i = pA->rank - 2; i >= 0; i--)
	{
		aStrides[i] = aStrides[i + 1] * pA->shape[i + 1];
	}

	for (int32_t i = 0; i < totalElements; i++)
	{
		int32_t aIdx = i;
		int32_t bIdx = 0;
		int32_t temp = i;
		for (int32_t j = 0; j < pB->rank; j++)
		{
			int32_t aRankOffset = pA->rank - pB->rank + j;
			int32_t dimIdx		= (temp / aStrides[aRankOffset]) % pA->shape[aRankOffset];
			bIdx += dimIdx * bStrides[j];
			temp %= aStrides[aRankOffset];
		}

		float aVal		 = __bfloat162float(pA->data[aIdx]) * alpha;
		float bVal		 = __bfloat162float(pB->data[bIdx]) * beta;
		pOutput->data[i] = __float2bfloat16(aVal + bVal);
	}

	free(aStrides);
	free(bStrides);

	*ppOutput = pOutput;
	return true;
}

bool lyTensorElementwiseMul(lyTensor** ppOutput, lyTensor* pA, lyTensor* pB)
{
	if (!ppOutput || !pA || !pB || !pA->data || !pB->data)
	{
		return false;
	}

	if (pA->rank != pB->rank)
	{
		return false;
	}

	for (int32_t i = 0; i < pA->rank; i++)
	{
		if (pA->shape[i] != pB->shape[i])
		{
			return false;
		}
	}

	lyTensor* pOutput;
	lyCreateTensor(&pOutput, pA->shape, pA->rank, NULL, NULL);

	int32_t totalElements = 1;
	for (int32_t i = 0; i < pA->rank; i++)
	{
		totalElements *= pA->shape[i];
	}

	for (int32_t i = 0; i < totalElements; i++)
	{
		float aVal		 = __bfloat162float(pA->data[i]);
		float bVal		 = __bfloat162float(pB->data[i]);
		pOutput->data[i] = __float2bfloat16(aVal * bVal);
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorMakeTriangularMask(lyTensor* pTensor)
{
	if (!pTensor || !pTensor->data || pTensor->rank != 2)
	{
		return false;
	}

	int32_t rows = pTensor->shape[0];
	int32_t cols = pTensor->shape[1];

	for (int32_t row = 0; row < rows; row++)
	{
		for (int32_t col = 0; col < cols; col++)
		{
			float val						= col <= row ? 0.0f : -HUGE_VALF;
			pTensor->data[row * cols + col] = __float2bfloat16(val);
		}
	}

	return true;
}

bool lyTensorArgmax(lyTensor** ppOutput, lyTensor* pInput, int32_t dim)
{
	if (!ppOutput || !pInput || dim < 0 || dim >= pInput->rank)
	{
		return false;
	}

	int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * (pInput->rank - 1));
	if (!newShape)
	{
		return false;
	}

	int32_t j			  = 0;
	int32_t totalElements = 1;
	for (int32_t i = 0; i < pInput->rank; i++)
	{
		if (i != dim)
		{
			newShape[j++] = pInput->shape[i];
			totalElements *= pInput->shape[i];
		}
	}

	lyTensor* pOutput;
	lyCreateTensor(&pOutput, newShape, pInput->rank - 1, NULL, NULL);
	free(newShape);

	int32_t* strides = (int32_t*)malloc(sizeof(int32_t) * pInput->rank);
	if (!strides)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	strides[pInput->rank - 1] = 1;
	for (int32_t i = pInput->rank - 2; i >= 0; i--)
	{
		strides[i] = strides[i + 1] * pInput->shape[i + 1];
	}

	int32_t dimSize = pInput->shape[dim];
	for (int32_t i = 0; i < totalElements; i++)
	{
		int32_t temp	 = i;
		int32_t inputIdx = 0;
		int32_t outDim	 = 0;

		for (int32_t d = 0; d < pInput->rank; d++)
		{
			if (d != dim)
			{
				int32_t idx = temp % pInput->shape[d];
				temp /= pInput->shape[d];
				inputIdx += idx * strides[d];
				outDim++;
			}
		}

		float	maxVal = -HUGE_VALF;
		int32_t maxIdx = 0;

		for (int32_t d = 0; d < dimSize; d++)
		{
			int32_t idx = inputIdx + d * strides[dim];
			float	val = __bfloat162float(pInput->data[idx]);
			if (val > maxVal)
			{
				maxVal = val;
				maxIdx = d;
			}
		}

		pOutput->data[i] = __float2bfloat16((float)maxIdx);
	}

	free(strides);
	*ppOutput = pOutput;
	return true;
}

bool lyTensorOuter(lyTensor** ppOutput, lyTensor* pA, lyTensor* pB)
{
	if (!ppOutput || !pA || !pB || !pA->data || !pB->data)
	{
		return false;
	}

	if (pA->rank != 1 || pB->rank != 1)
	{
		return false;
	}

	lyTensor* pOutput;
	int32_t	  outputShape[] = {pA->shape[0], pB->shape[0]};
	lyCreateTensor(&pOutput, outputShape, 2, NULL, NULL);

	for (int32_t i = 0; i < pA->shape[0]; i++)
	{
		for (int32_t j = 0; j < pB->shape[0]; j++)
		{
			float aVal							= __bfloat162float(pA->data[i]);
			float bVal							= __bfloat162float(pB->data[j]);
			pOutput->data[i * pB->shape[0] + j] = __float2bfloat16(aVal * bVal);
		}
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorEmbedding(lyTensor** ppOutput, lyTensor* pTokens, lyTensor* pEmbeddings)
{
	if (!ppOutput || !pTokens || !pEmbeddings || pTokens->rank != 1 || pEmbeddings->rank != 2)
	{
		return false;
	}

	int32_t seqLen = pTokens->shape[0];
	int32_t dim	   = pEmbeddings->shape[1];

	lyTensor* pOutput;
	int32_t	  outputShape[] = {seqLen, dim};
	lyCreateTensor(&pOutput, outputShape, 2, NULL, NULL);

	for (int32_t i = 0; i < seqLen; i++)
	{
		float	tokenVal = __bfloat162float(pTokens->data[i]);
		int32_t tokenId	 = (int32_t)tokenVal;

		if (tokenId < 0)
		{
			for (int32_t j = 0; j < dim; j++)
			{
				pOutput->data[i * dim + j] = __float2bfloat16(0.0f);
			}
			continue;
		}

		for (int32_t j = 0; j < dim; j++)
		{
			pOutput->data[i * dim + j] = pEmbeddings->data[tokenId * dim + j];
		}
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorTranspose(lyTensor** ppOutput, lyTensor* pInput, int32_t* pPerm)
{
	if (!ppOutput || !pInput || !pPerm || pInput->rank < 2)
	{
		return false;
	}

	int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * pInput->rank);
	if (!newShape)
	{
		return false;
	}

	for (int32_t i = 0; i < pInput->rank; i++)
	{
		newShape[i] = pInput->shape[pPerm[i]];
	}

	lyTensor* pOutput;
	lyCreateTensor(&pOutput, newShape, pInput->rank, NULL, NULL);
	free(newShape);

	if (pInput->rank == 2)
	{
		int32_t rows = pInput->shape[0];
		int32_t cols = pInput->shape[1];

		for (int32_t i = 0; i < rows; i++)
		{
			for (int32_t j = 0; j < cols; j++)
			{
				int32_t inIdx = i * cols + j;
				int32_t outIdx;

				if (pPerm[0] == 0 && pPerm[1] == 1)
				{
					outIdx = inIdx;
				}
				else if (pPerm[0] == 1 && pPerm[1] == 0)
				{
					outIdx = j * rows + i;
				}
				else
				{
					return false;
				}

				pOutput->data[outIdx] = pInput->data[inIdx];
			}
		}
	}
	else if (pInput->rank == 3)
	{
		int32_t inputStrides[3]	 = {0};
		int32_t outputStrides[3] = {0};

		inputStrides[pInput->rank - 1] = 1;
		for (int32_t i = pInput->rank - 2; i >= 0; i--)
		{
			inputStrides[i] = inputStrides[i + 1] * pInput->shape[i + 1];
		}

		outputStrides[pInput->rank - 1] = 1;
		for (int32_t i = pInput->rank - 2; i >= 0; i--)
		{
			outputStrides[i] = outputStrides[i + 1] * pOutput->shape[i + 1];
		}

		int32_t indices[3] = {0};
		int32_t inputIdx = 0, outputIdx = 0;

		for (int32_t i = 0; i < pInput->shape[0]; i++)	// seqLen
		{
			for (int32_t j = 0; j < pInput->shape[1]; j++)	// nKVHeads
			{
				for (int32_t k = 0; k < pInput->shape[2]; k++)	// headDim
				{
					indices[0] = i;
					indices[1] = j;
					indices[2] = k;

					inputIdx  = indices[0] * inputStrides[0] + indices[1] * inputStrides[1] + indices[2] * inputStrides[2];
					outputIdx = indices[pPerm[0]] * outputStrides[0] + indices[pPerm[1]] * outputStrides[1] + indices[pPerm[2]] * outputStrides[2];

					pOutput->data[outputIdx] = pInput->data[inputIdx];
				}
			}
		}
	}

	*ppOutput = pOutput;
	return true;
}