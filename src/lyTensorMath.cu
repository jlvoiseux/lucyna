#include "lyTensorMath.h"

#include <cuda_bf16.h>

#define LY_MAX_RANK 4

static int lyTensorGetElementCount(const lyTensor* pTensor)
{
	int count = 1;
	for (int i = 0; i < pTensor->rank; i++)
	{
		count *= pTensor->shape[i];
	}
	return count;
}

static int32_t getTotalSize(const int32_t* shape, int32_t rank)
{
	int32_t size = 1;
	for (int32_t i = 0; i < rank; i++)
	{
		size *= shape[i];
	}
	return size;
}

void calculateStrides(int32_t* strides, const int32_t* shape, int32_t rank)
{
	strides[rank - 1] = 1;
	for (int32_t i = rank - 2; i >= 0; i--)
	{
		strides[i] = strides[i + 1] * shape[i + 1];
	}
}

__global__ void tensorAddKernel(nv_bfloat16* output, const nv_bfloat16* a, const nv_bfloat16* b, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	output[idx] = __hadd(a[idx], b[idx]);
}

__global__ void tensorMatMulKernel(nv_bfloat16* output, const nv_bfloat16* a, const nv_bfloat16* b, const int32_t* aStrides, const int32_t* bStrides, const int32_t* outStrides, const int32_t* aShape, const int32_t* bShape, int32_t rank, int32_t batchSize, int32_t m, int32_t n, int32_t k)
{
	int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batchSize * m * n)
		return;

	int32_t batchIdx = idx / (m * n);
	int32_t row		 = (idx % (m * n)) / n;
	int32_t col		 = idx % n;

	int32_t aOffset = 0;
	int32_t bOffset = 0;
	for (int32_t i = 0; i < rank - 2; i++)
	{
		int32_t dim = batchIdx / aStrides[i];
		aOffset += dim * aShape[i];
		bOffset += dim * bShape[i];
		batchIdx %= aStrides[i];
	}

	nv_bfloat16 sum = __float2bfloat16(0.0f);
	for (int32_t i = 0; i < k; i++)
	{
		int32_t aIdx = aOffset + row * k + i;
		int32_t bIdx = bOffset + i * n + col;
		sum			 = __hadd(sum, __hmul(a[aIdx], b[bIdx]));
	}
	output[idx] = sum;
}

__global__ void tensorScaleAndAddKernel(nv_bfloat16* output, const nv_bfloat16* input, const nv_bfloat16* mask, float scale, int numRows, int numCols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < numRows && col < numCols)
	{
		int idx		= row * numCols + col;
		output[idx] = __hmul(input[idx], __float2bfloat16(scale));
		if (mask)
		{
			output[idx] = __hadd(output[idx], mask[idx]);
		}
	}
}

__global__ void tensorElementwiseMulKernel(nv_bfloat16* output, const nv_bfloat16* a, const nv_bfloat16* b, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	output[idx] = __hmul(a[idx], b[idx]);
}

__global__ void triangularMaskKernel(nv_bfloat16* output, int32_t rows, int32_t cols)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < rows && col < cols)
	{
		float val				 = col <= row ? 0.0f : -INFINITY;
		output[row * cols + col] = __float2bfloat16(val);
	}
}

__global__ void tensorArgmaxKernel(nv_bfloat16* output, const nv_bfloat16* input, int32_t batchSize, int32_t dimSize)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= batchSize)
	{
		return;
	}

	float	maxVal = -INFINITY;
	int32_t maxIdx = 0;

	for (int32_t i = 0; i < dimSize; i++)
	{
		float val = __bfloat162float(input[idx * dimSize + i]);
		if (val > maxVal)
		{
			maxVal = val;
			maxIdx = i;
		}
	}

	output[idx] = __float2bfloat16((float)maxIdx);
}

__global__ void tensorOuterKernel(nv_bfloat16* output, const nv_bfloat16* a, const nv_bfloat16* b, int aSize, int bSize)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < aSize && col < bSize)
	{
		output[row * bSize + col] = __hmul(a[row], b[col]);
	}
}

__global__ void tensorEmbeddingKernel(nv_bfloat16* output, const nv_bfloat16* tokens, const nv_bfloat16* embeddings, int seqLen, int dim)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= seqLen * dim)
		return;

	int seqPos = idx / dim;
	int dimPos = idx % dim;

	int tokenId = (int)__bfloat162float(tokens[seqPos]);
	output[idx] = embeddings[tokenId * dim + dimPos];
}

__global__ void tensorTransposeKernel(nv_bfloat16* output, const nv_bfloat16* input, const int32_t* inputShape, const int32_t* inputStrides, const int32_t* outputStrides, const int32_t* perm, int32_t rank, int32_t elementCount)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= elementCount)
		return;

	int32_t inputIdx				 = idx;
	int32_t multiDimIdx[LY_MAX_RANK] = {0};

	for (int32_t i = rank - 1; i >= 0; i--)
	{
		multiDimIdx[i] = inputIdx / inputStrides[i];
		inputIdx %= inputStrides[i];
	}

	int32_t outputIdx = 0;
	for (int32_t i = 0; i < rank; i++)
	{
		outputIdx += multiDimIdx[perm[i]] * outputStrides[i];
	}

	output[outputIdx] = input[idx];
}

bool lyTensorAdd(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB)
{
	if (!ppOutput || !pA || !pB || !pA->data || !pB->data)
	{
		return false;
	}

	if (pA->rank != pB->rank)
	{
		return false;
	}
	for (int i = 0; i < pA->rank; i++)
	{
		if (pA->shape[i] != pB->shape[i])
		{
			return false;
		}
	}

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	if (!lySetTensorShape(pOutput, pA->shape, pA->rank) || !lySetTensorData(pOutput, NULL, lyTensorGetElementCount(pOutput) * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	int size	  = lyTensorGetElementCount(pOutput);
	int blockSize = 256;
	int gridSize  = (size + blockSize - 1) / blockSize;

	tensorAddKernel<<<gridSize, blockSize>>>(pOutput->data, pA->data, pB->data, size);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorMatMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB)
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

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
		return false;

	int32_t* outShape = (int32_t*)malloc(sizeof(int32_t) * pA->rank);
	if (!outShape)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	for (int32_t i = 0; i < pA->rank - 2; i++)
	{
		outShape[i] = pA->shape[i];
	}
	outShape[pA->rank - 2] = m;
	outShape[pA->rank - 1] = n;

	if (!lySetTensorShape(pOutput, outShape, pA->rank) || !lySetTensorData(pOutput, NULL, getTotalSize(outShape, pA->rank) * sizeof(nv_bfloat16)))
	{
		free(outShape);
		lyDestroyTensor(pOutput);
		return false;
	}
	free(outShape);

	int32_t* aStrides	= (int32_t*)malloc(sizeof(int32_t) * pA->rank);
	int32_t* bStrides	= (int32_t*)malloc(sizeof(int32_t) * pB->rank);
	int32_t* outStrides = (int32_t*)malloc(sizeof(int32_t) * pA->rank);
	if (!aStrides || !bStrides || !outStrides)
	{
		free(aStrides);
		free(bStrides);
		free(outStrides);
		lyDestroyTensor(pOutput);
		return false;
	}

	int32_t *dAShape, *dBShape, *dAStrides, *dBStrides, *dOutStrides;
	cudaMalloc(&dAShape, sizeof(int32_t) * pA->rank);
	cudaMalloc(&dBShape, sizeof(int32_t) * pB->rank);
	cudaMalloc(&dAStrides, sizeof(int32_t) * pA->rank);
	cudaMalloc(&dBStrides, sizeof(int32_t) * pB->rank);
	cudaMalloc(&dOutStrides, sizeof(int32_t) * pA->rank);

	cudaMemcpy(dAShape, pA->shape, sizeof(int32_t) * pA->rank, cudaMemcpyHostToDevice);
	cudaMemcpy(dBShape, pB->shape, sizeof(int32_t) * pB->rank, cudaMemcpyHostToDevice);
	cudaMemcpy(dAStrides, aStrides, sizeof(int32_t) * pA->rank, cudaMemcpyHostToDevice);
	cudaMemcpy(dBStrides, bStrides, sizeof(int32_t) * pB->rank, cudaMemcpyHostToDevice);
	cudaMemcpy(dOutStrides, outStrides, sizeof(int32_t) * pA->rank, cudaMemcpyHostToDevice);

	int32_t totalElements = batchSize * m * n;
	int32_t blockSize	  = 256;
	int32_t gridSize	  = (totalElements + blockSize - 1) / blockSize;

	tensorMatMulKernel<<<gridSize, blockSize>>>(pOutput->data, pA->data, pB->data, dAStrides, dBStrides, dOutStrides, dAShape, dBShape, pA->rank, batchSize, m, n, k);

	cudaFree(dAShape);
	cudaFree(dBShape);
	cudaFree(dAStrides);
	cudaFree(dBStrides);
	cudaFree(dOutStrides);
	free(aStrides);
	free(bStrides);
	free(outStrides);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorScaleAndAdd(lyTensor** ppOutput, lyTensor* pInput, const lyTensor* pMask, float scale)
{
	if (!ppOutput || !pInput || !pInput->data)
	{
		return false;
	}

	if (pInput->rank != 2)
	{
		return false;
	}

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	if (!lySetTensorShape(pOutput, pInput->shape, pInput->rank) || !lySetTensorData(pOutput, NULL, lyTensorGetElementCount(pOutput) * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	dim3 blockSize(16, 16);
	dim3 gridSize((pInput->shape[1] + blockSize.x - 1) / blockSize.x, (pInput->shape[0] + blockSize.y - 1) / blockSize.y);

	tensorScaleAndAddKernel<<<gridSize, blockSize>>>(pOutput->data, pInput->data, pMask ? pMask->data : nullptr, scale, pInput->shape[0], pInput->shape[1]);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorElementwiseMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB)
{
	if (!ppOutput || !pA || !pB || !pA->data || !pB->data)
	{
		return false;
	}

	if (pA->rank != pB->rank)
	{
		return false;
	}
	for (int i = 0; i < pA->rank; i++)
	{
		if (pA->shape[i] != pB->shape[i])
		{
			return false;
		}
	}

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	if (!lySetTensorShape(pOutput, pA->shape, pA->rank) || !lySetTensorData(pOutput, NULL, lyTensorGetElementCount(pOutput) * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	int totalElements = 1;
	for (int i = 0; i < pA->rank; i++)
	{
		totalElements *= pA->shape[i];
	}

	int blockSize = 256;
	int gridSize  = (totalElements + blockSize - 1) / blockSize;

	tensorElementwiseMulKernel<<<gridSize, blockSize>>>(pOutput->data, pA->data, pB->data, totalElements);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
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

	dim3 blockSize(16, 16);
	dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

	triangularMaskKernel<<<gridSize, blockSize>>>(pTensor->data, rows, cols);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		return false;
	}

	return true;
}

bool lyTensorArgmax(lyTensor** ppOutput, const lyTensor* pInput, int32_t dim)
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

	int32_t batchSize = 1;
	int32_t j		  = 0;
	for (int32_t i = 0; i < pInput->rank; i++)
	{
		if (i != dim)
		{
			newShape[j++] = pInput->shape[i];
			batchSize *= pInput->shape[i];
		}
	}

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		free(newShape);
		return false;
	}

	if (!lySetTensorShape(pOutput, newShape, pInput->rank - 1) || !lySetTensorData(pOutput, NULL, lyTensorGetElementCount(pOutput) * sizeof(nv_bfloat16)))
	{
		free(newShape);
		lyDestroyTensor(pOutput);
		return false;
	}
	free(newShape);

	int32_t dimSize	  = pInput->shape[dim];
	int32_t blockSize = 256;
	int32_t numBlocks = (batchSize + blockSize - 1) / blockSize;

	tensorArgmaxKernel<<<numBlocks, blockSize>>>(pOutput->data, pInput->data, batchSize, dimSize);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorOuter(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB)
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
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	int32_t outputShape[] = {pA->shape[0], pB->shape[0]};
	if (!lySetTensorShape(pOutput, outputShape, 2) || !lySetTensorData(pOutput, NULL, lyTensorGetElementCount(pOutput) * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	dim3 blockSize(16, 16);
	dim3 gridSize((pB->shape[0] + blockSize.x - 1) / blockSize.x, (pA->shape[0] + blockSize.y - 1) / blockSize.y);

	tensorOuterKernel<<<gridSize, blockSize>>>(pOutput->data, pA->data, pB->data, pA->shape[0], pB->shape[0]);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorEmbedding(lyTensor** ppOutput, const lyTensor* pTokens, const lyTensor* pEmbeddings)
{
	if (!ppOutput || !pTokens || !pEmbeddings || pTokens->rank != 1 || pEmbeddings->rank != 2)
	{
		return false;
	}

	int seqLen = pTokens->shape[0];
	int dim	   = pEmbeddings->shape[1];

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	int32_t outputShape[] = {seqLen, dim};
	if (!lySetTensorShape(pOutput, outputShape, 2) || !lySetTensorData(pOutput, NULL, seqLen * dim * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	int totalElements = seqLen * dim;
	int blockSize	  = 256;
	int numBlocks	  = (totalElements + blockSize - 1) / blockSize;

	tensorEmbeddingKernel<<<numBlocks, blockSize>>>(pOutput->data, pTokens->data, pEmbeddings->data, seqLen, dim);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}

bool lyTensorTranspose(lyTensor** ppOutput, const lyTensor* pInput, const int32_t* perm)
{
	if (!ppOutput || !pInput || !pInput->data || !perm || pInput->rank < 2)
	{
		return false;
	}

	int32_t rank		 = pInput->rank;
	int32_t elementCount = lyTensorGetElementCount(pInput);

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	int32_t* outputShape = (int32_t*)malloc(rank * sizeof(int32_t));
	if (!outputShape)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	for (int32_t i = 0; i < rank; i++)
	{
		outputShape[i] = pInput->shape[perm[i]];
	}

	if (!lySetTensorShape(pOutput, outputShape, rank) || !lySetTensorData(pOutput, NULL, elementCount * sizeof(nv_bfloat16)))
	{
		free(outputShape);
		lyDestroyTensor(pOutput);
		return false;
	}
	free(outputShape);

	int32_t* inputStrides  = (int32_t*)malloc(rank * sizeof(int32_t));
	int32_t* outputStrides = (int32_t*)malloc(rank * sizeof(int32_t));
	if (!inputStrides || !outputStrides)
	{
		free(inputStrides);
		free(outputStrides);
		lyDestroyTensor(pOutput);
		return false;
	}

	calculateStrides(inputStrides, pInput->shape, rank);
	calculateStrides(outputStrides, pOutput->shape, rank);

	int32_t *dInputShape, *dInputStrides, *dOutputStrides, *dPerm;
	cudaMalloc(&dInputShape, sizeof(int32_t) * rank);
	cudaMalloc(&dInputStrides, sizeof(int32_t) * rank);
	cudaMalloc(&dOutputStrides, sizeof(int32_t) * rank);
	cudaMalloc(&dPerm, sizeof(int32_t) * rank);

	cudaMemcpy(dInputShape, pInput->shape, sizeof(int32_t) * rank, cudaMemcpyHostToDevice);
	cudaMemcpy(dInputStrides, inputStrides, sizeof(int32_t) * rank, cudaMemcpyHostToDevice);
	cudaMemcpy(dOutputStrides, outputStrides, sizeof(int32_t) * rank, cudaMemcpyHostToDevice);
	cudaMemcpy(dPerm, perm, sizeof(int32_t) * rank, cudaMemcpyHostToDevice);

	int32_t blockSize = 256;
	int32_t gridSize  = (elementCount + blockSize - 1) / blockSize;

	tensorTransposeKernel<<<gridSize, blockSize>>>(pOutput->data, pInput->data, dInputShape, dInputStrides, dOutputStrides, dPerm, rank, elementCount);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		cudaFree(dInputShape);
		cudaFree(dInputStrides);
		cudaFree(dOutputStrides);
		cudaFree(dPerm);
		free(inputStrides);
		free(outputStrides);
		lyDestroyTensor(pOutput);
		return false;
	}

	cudaFree(dInputShape);
	cudaFree(dInputStrides);
	cudaFree(dOutputStrides);
	cudaFree(dPerm);
	free(inputStrides);
	free(outputStrides);

	*ppOutput = pOutput;
	return true;
}