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

__global__ void tensorScaleKernel(nv_bfloat16* output, const nv_bfloat16* input, float (*scaleFn)(float), int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
	{
		return;
	}

	float val	= __bfloat162float(input[idx]);
	output[idx] = __float2bfloat16(scaleFn(val));
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

	if (!lySetTensorShape(pOutput, pA->shape, pA->rank))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	if (pOutput->memoryType != LY_MEMORY_GPU)
	{
		if (!lyTensorToGPU(pOutput))
		{
			lyDestroyTensor(pOutput);
			return false;
		}
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
	if (!lySetTensorShape(pOutput, outputShape, 2))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	if (pOutput->memoryType != LY_MEMORY_GPU)
	{
		if (!lyTensorToGPU(pOutput))
		{
			lyDestroyTensor(pOutput);
			return false;
		}
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

	if (!lySetTensorShape(pOutput, pInput->shape, pInput->rank))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	if (pOutput->memoryType != LY_MEMORY_GPU)
	{
		if (!lyTensorToGPU(pOutput))
		{
			lyDestroyTensor(pOutput);
			return false;
		}
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

	if (!lySetTensorShape(pOutput, pA->shape, pA->rank))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	if (pOutput->memoryType != LY_MEMORY_GPU)
	{
		if (!lyTensorToGPU(pOutput))
		{
			lyDestroyTensor(pOutput);
			return false;
		}
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

bool lyTensorScale(lyTensor** ppOutput, const lyTensor* pInput, float (*scaleFn)(float input))
{
	if (!ppOutput || !pInput || !scaleFn)
	{
		return false;
	}

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	if (!lySetTensorShape(pOutput, pInput->shape, pInput->rank))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	int totalSize = 1;
	for (int i = 0; i < pInput->rank; i++)
	{
		totalSize *= pInput->shape[i];
	}

	int blockSize = 256;
	int gridSize  = (totalSize + blockSize - 1) / blockSize;

	tensorScaleKernel<<<gridSize, blockSize>>>(pOutput->data, pInput->data, scaleFn, totalSize);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}