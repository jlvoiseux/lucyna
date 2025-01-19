#include "lyTensorMath.h"

#include <cuda_bf16.h>

static int lyTensorGetElementCount(const lyTensor* pTensor)
{
	int count = 1;
	for (int i = 0; i < pTensor->rank; i++)
	{
		count *= pTensor->shape[i];
	}
	return count;
}

__global__ void tensorAddKernel(nv_bfloat16* output, const nv_bfloat16* a, const nv_bfloat16* b, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	output[idx] = __hadd(a[idx], b[idx]);
}

__global__ void tensorMatMulKernel(nv_bfloat16* output, const nv_bfloat16* a, const nv_bfloat16* b, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m && col < n)
	{
		nv_bfloat16 sum = __float2bfloat16(0.0f);
		for (int i = 0; i < k; ++i)
		{
			sum = __hadd(sum, __hmul(a[row * k + i], b[i * n + col]));
		}
		output[row * n + col] = sum;
	}
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
	if (!ppOutput || !pA || !pB || !pA->data || !pB->data)
	{
		return false;
	}

	if (pA->rank != 2 || pB->rank != 2 || pA->shape[1] != pB->shape[0])
	{
		return false;
	}

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	int32_t outputShape[] = {pA->shape[0], pB->shape[1]};
	if (!lySetTensorShape(pOutput, outputShape, 2) || !lySetTensorData(pOutput, NULL, lyTensorGetElementCount(pOutput) * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	dim3 blockSize(16, 16);
	dim3 gridSize((pB->shape[1] + blockSize.x - 1) / blockSize.x, (pA->shape[0] + blockSize.y - 1) / blockSize.y);

	tensorMatMulKernel<<<gridSize, blockSize>>>(pOutput->data, pA->data, pB->data, pA->shape[0], pB->shape[1], pA->shape[1]);

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