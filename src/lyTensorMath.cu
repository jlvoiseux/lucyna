#include "lyTensorMath.h"

#include <cuda_bf16.h>
#include <float.h>
#include <math_constants.h>
#include <stdio.h>

__global__ void matMulKernel(const nv_bfloat16* A, const nv_bfloat16* B, float* C, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
	{
		float sum = 0.0f;
		for (int i = 0; i < k; i++)
		{
			float aVal = __bfloat162float(A[row * k + i]);
			float bVal = __bfloat162float(B[i * n + col]);
			sum += aVal * bVal;
		}
		C[row * n + col] = sum;
	}
}

void lyTensorMatMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB)
{
	int32_t m = pA->shape[pA->rank - 2];
	int32_t k = pA->shape[pA->rank - 1];
	int32_t n = pB->shape[pB->rank - 1];

	int32_t batchSize = 1;
	for (int32_t i = 0; i < pA->rank - 2; i++)
	{
		batchSize *= pA->shape[i];
	}

	int32_t* outShape = (int32_t*)malloc(sizeof(int32_t) * pA->rank);
	for (int32_t i = 0; i < pA->rank - 2; i++)
	{
		outShape[i] = pA->shape[i];
	}
	outShape[pA->rank - 2] = m;
	outShape[pA->rank - 1] = n;

	lyTensor* pOutput;
	lyTensorCreate(&pOutput, outShape, pA->rank, NULL, NULL);
	free(outShape);

	size_t matrixSizeA = m * k * sizeof(nv_bfloat16);
	size_t matrixSizeB = k * n * sizeof(nv_bfloat16);
	size_t matrixSizeC = m * n * sizeof(float);	 // Float intermediate storage

	nv_bfloat16 *d_A, *d_B;
	float*		 d_C;
	cudaMalloc(&d_A, matrixSizeA);
	cudaMalloc(&d_B, matrixSizeB);
	cudaMalloc(&d_C, matrixSizeC);

	float* tempOutput = (float*)malloc(matrixSizeC);

	for (int32_t batch = 0; batch < batchSize; batch++)
	{
		nv_bfloat16* pBatchA   = pA->data + batch * m * k;
		nv_bfloat16* pBatchB   = pB->data + batch * k * n;
		nv_bfloat16* pBatchOut = pOutput->data + batch * m * n;

		cudaMemcpy(d_A, pBatchA, matrixSizeA, cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, pBatchB, matrixSizeB, cudaMemcpyHostToDevice);

		dim3 blockDim(16, 16);
		dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
		matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, n, k);

		cudaMemcpy(tempOutput, d_C, matrixSizeC, cudaMemcpyDeviceToHost);

		for (int i = 0; i < m * n; i++)
		{
			pBatchOut[i] = __float2bfloat16_rz(tempOutput[i]);
		}
	}

	free(tempOutput);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	*ppOutput = pOutput;
}

__global__ void scaleAndAddKernel(const nv_bfloat16* A, const nv_bfloat16* B, nv_bfloat16* C, const int32_t* aStrides, const int32_t* bStrides, const int32_t* aShape, int32_t aRank, int32_t bRank, float alpha, float beta, int32_t totalElements)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= totalElements)
		return;

	float aVal = __bfloat162float(A[idx]) * alpha;

	if (B != NULL)
	{
		int32_t bIdx = 0;
		int32_t temp = idx;

		for (int32_t j = 0; j < bRank; j++)
		{
			int32_t aRankOffset = aRank - bRank + j;
			int32_t dimIdx		= (temp / aStrides[aRankOffset]) % aShape[aRankOffset];
			bIdx += dimIdx * bStrides[j];
			temp %= aStrides[aRankOffset];
		}

		C[idx] = __float2bfloat16_rz(aVal + __bfloat162float(B[bIdx]) * beta);
	}
	else
	{
		C[idx] = __float2bfloat16_rz(aVal);
	}
}

void lyTensorScaleAndAdd(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, nv_bfloat16 alpha, nv_bfloat16 beta)
{
	lyTensor* pOutput;
	lyTensorCreate(&pOutput, pA->shape, pA->rank, NULL, NULL);

	int32_t totalElements = 1;
	for (int32_t i = 0; i < pA->rank; i++)
	{
		totalElements *= pA->shape[i];
	}

	int32_t* aStrides = (int32_t*)malloc(pA->rank * sizeof(int32_t));
	int32_t* bStrides = NULL;

	aStrides[pA->rank - 1] = 1;
	for (int32_t i = pA->rank - 2; i >= 0; i--)
	{
		aStrides[i] = aStrides[i + 1] * pA->shape[i + 1];
	}

	if (pB)
	{
		bStrides			   = (int32_t*)malloc(pB->rank * sizeof(int32_t));
		bStrides[pB->rank - 1] = 1;
		for (int32_t i = pB->rank - 2; i >= 0; i--)
		{
			bStrides[i] = bStrides[i + 1] * pB->shape[i + 1];
		}
	}

	nv_bfloat16 *d_A, *d_B				 = NULL, *d_C;
	int32_t *	 d_aStrides, *d_bStrides = NULL, *d_aShape;

	size_t dataSizeA = totalElements * sizeof(nv_bfloat16);

	cudaMalloc(&d_A, dataSizeA);
	cudaMalloc(&d_C, dataSizeA);
	cudaMalloc(&d_aStrides, pA->rank * sizeof(int32_t));
	cudaMalloc(&d_aShape, pA->rank * sizeof(int32_t));
	cudaMemcpy(d_A, pA->data, dataSizeA, cudaMemcpyHostToDevice);
	cudaMemcpy(d_aStrides, aStrides, pA->rank * sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(d_aShape, pA->shape, pA->rank * sizeof(int32_t), cudaMemcpyHostToDevice);

	size_t totalElementsB = 1;
	if (pB)
	{
		for (int32_t i = 0; i < pB->rank; i++)
		{
			totalElementsB *= pB->shape[i];
		}
		size_t dataSizeB = totalElementsB * sizeof(nv_bfloat16);
		cudaMalloc(&d_B, dataSizeB);
		cudaMalloc(&d_bStrides, pB->rank * sizeof(int32_t));
		cudaMemcpy(d_B, pB->data, dataSizeB, cudaMemcpyHostToDevice);
		cudaMemcpy(d_bStrides, bStrides, pB->rank * sizeof(int32_t), cudaMemcpyHostToDevice);
	}

	int threadsPerBlock = 256;
	int numBlocks		= (totalElements + threadsPerBlock - 1) / threadsPerBlock;

	float alphaF = __bfloat162float(alpha);
	float betaF	 = __bfloat162float(beta);
	scaleAndAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, d_aStrides, d_bStrides, d_aShape, pA->rank, pB ? pB->rank : 0, alphaF, betaF, totalElements);

	cudaMemcpy(pOutput->data, d_C, dataSizeA, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_C);
	cudaFree(d_aStrides);
	cudaFree(d_aShape);

	if (pB)
	{
		cudaFree(d_B);
		cudaFree(d_bStrides);
		free(bStrides);
	}

	free(aStrides);

	*ppOutput = pOutput;
}

void lyTensorElementwiseMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB)
{
	lyTensor* pOutput;
	lyTensorCreate(&pOutput, pA->shape, pA->rank, NULL, NULL);

	int32_t* aStrides = (int32_t*)malloc(pA->rank * sizeof(int32_t));
	int32_t* bStrides = (int32_t*)malloc(pB->rank * sizeof(int32_t));

	aStrides[pA->rank - 1] = 1;
	for (int32_t i = pA->rank - 2; i >= 0; i--)
	{
		aStrides[i] = aStrides[i + 1] * pA->shape[i + 1];
	}

	bStrides[pB->rank - 1] = 1;
	for (int32_t i = pB->rank - 2; i >= 0; i--)
	{
		bStrides[i] = bStrides[i + 1] * pB->shape[i + 1];
	}

	int32_t totalElements = 1;
	for (int32_t i = 0; i < pA->rank; i++)
	{
		totalElements *= pA->shape[i];
	}

	for (int32_t i = 0; i < totalElements; i++)
	{
		int32_t bIdx = 0;
		int32_t temp = i;

		for (int32_t j = 0; j < pB->rank; j++)
		{
			int32_t aRankOffset = pA->rank - pB->rank + j;
			int32_t dimIdx		= (temp / aStrides[aRankOffset]) % pA->shape[aRankOffset];
			bIdx += dimIdx * bStrides[j];
			temp %= aStrides[aRankOffset];
		}

		pOutput->data[i] = __float2bfloat16_rz(__bfloat162float(pA->data[i]) * __bfloat162float(pB->data[bIdx]));
	}

	free(aStrides);
	free(bStrides);

	*ppOutput = pOutput;
}

void lyTensorMakeTriangularMask(lyTensor* pTensor)
{
	int32_t rows = pTensor->shape[0];
	int32_t cols = pTensor->shape[1];

	for (int32_t row = 0; row < rows; row++)
	{
		for (int32_t col = 0; col < cols; col++)
		{
			float val						= col <= row ? 0.0f : -HUGE_VALF;
			pTensor->data[row * cols + col] = __float2bfloat16_rz(val);
		}
	}
}

void lyTensorSoftmax(lyTensor** ppOutput, const lyTensor* pInput)
{
	int32_t	  dim = pInput->rank - 1;
	lyTensor* pOutput;
	lyTensorCreate(&pOutput, pInput->shape, pInput->rank, NULL, NULL);

	int32_t dimSize	  = pInput->shape[dim];
	int32_t outerSize = 1;
	for (int32_t i = 0; i < dim; i++)
	{
		outerSize *= pInput->shape[i];
	}

	for (int32_t i = 0; i < outerSize; i++)
	{
		int32_t rowOffset = i * dimSize;

		float rowMax = -FLT_MAX;
		for (int32_t j = 0; j < dimSize; j++)
		{
			float val = __bfloat162float(pInput->data[rowOffset + j]);
			if (val > rowMax)
				rowMax = val;
		}

		float rowExpSum = 0.0f;
		for (int32_t j = 0; j < dimSize; j++)
		{
			float val = __bfloat162float(pInput->data[rowOffset + j]);
			rowExpSum += expf(val - rowMax);
		}

		for (int32_t j = 0; j < dimSize; j++)
		{
			float val					 = __bfloat162float(pInput->data[rowOffset + j]);
			float outVal				 = expf(val - rowMax) / rowExpSum;
			pOutput->data[rowOffset + j] = __float2bfloat16_rz(outVal);
		}
	}

	*ppOutput = pOutput;
}

void lyTensorArgmax(int32_t* pOutput, const lyTensor* pInput)
{
	if (pInput->rank > 2 || (pInput->rank == 2 && pInput->shape[0] != 1 && pInput->shape[1] != 1))
	{
		*pOutput = -1;
		return;
	}

	int32_t vectorLength = pInput->rank == 1 ? pInput->shape[0] : (pInput->shape[0] == 1 ? pInput->shape[1] : pInput->shape[0]);

	float	maxVal = -HUGE_VALF;
	int32_t maxIdx = 0;

	for (int32_t i = 0; i < vectorLength; i++)
	{
		float val = __bfloat162float(pInput->data[i]);
		if (val > maxVal)
		{
			maxVal = val;
			maxIdx = i;
		}
	}

	*pOutput = maxIdx;
}

void lyTensorOuter(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB)
{
	lyTensor* pOutput;
	int32_t	  outputShape[] = {pA->shape[0], pB->shape[0]};
	lyTensorCreate(&pOutput, outputShape, 2, NULL, NULL);

	for (int32_t i = 0; i < pA->shape[0]; i++)
	{
		for (int32_t j = 0; j < pB->shape[0]; j++)
		{
			float aVal							= __bfloat162float(pA->data[i]);
			float bVal							= __bfloat162float(pB->data[j]);
			pOutput->data[i * pB->shape[0] + j] = __float2bfloat16_rz(aVal * bVal);
		}
	}

	*ppOutput = pOutput;
}

void lyTensorEmbedding(lyTensor** ppOutput, const int32_t* pInputTokens, int32_t seqLen, const lyTensor* pEmbeddings)
{
	if (pEmbeddings->rank != 2)
	{
		return;
	}

	int32_t	  embeddingDim = pEmbeddings->shape[1];
	lyTensor* pOutput;
	int32_t	  outputShape[] = {seqLen, embeddingDim};
	lyTensorCreate(&pOutput, outputShape, 2, NULL, NULL);
	size_t rowSizeBytes = embeddingDim * sizeof(nv_bfloat16);

	for (int32_t rowIdx = 0; rowIdx < seqLen; rowIdx++)
	{
		int32_t tokenId	  = pInputTokens[rowIdx];
		size_t	srcOffset = tokenId * embeddingDim;
		size_t	dstOffset = rowIdx * embeddingDim;
		memcpy(pOutput->data + dstOffset, pEmbeddings->data + srcOffset, rowSizeBytes);
	}

	*ppOutput = pOutput;
}

void lyTensorTranspose(lyTensor** ppOutput, const lyTensor* pInput, const int32_t* pPerm)
{
	int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * pInput->rank);
	for (int32_t i = 0; i < pInput->rank; i++)
	{
		newShape[i] = pInput->shape[pPerm[i]];
	}

	lyTensor* pOutput;
	lyTensorCreate(&pOutput, newShape, pInput->rank, NULL, NULL);
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
					outIdx = inIdx;
				else if (pPerm[0] == 1 && pPerm[1] == 0)
					outIdx = j * rows + i;
				else
					return;

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

		for (int32_t i = 0; i < pInput->shape[0]; i++)
		{
			for (int32_t j = 0; j < pInput->shape[1]; j++)
			{
				for (int32_t k = 0; k < pInput->shape[2]; k++)
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
}