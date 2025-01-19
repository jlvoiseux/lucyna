#include "lyRMSNorm.h"

#include <cuda_bf16.h>
#include <stdlib.h>

bool lyCreateRMSNorm(lyRMSNorm** ppNorm, float epsilon, lyTensor* pWeights)
{
	if (!ppNorm || !pWeights)
	{
		return false;
	}

	lyRMSNorm* pNorm = (lyRMSNorm*)malloc(sizeof(lyRMSNorm));
	if (!pNorm)
	{
		return false;
	}

	pNorm->epsilon = epsilon;
	pNorm->weights = pWeights;
	*ppNorm		   = pNorm;

	return true;
}

void lyDestroyRMSNorm(lyRMSNorm* pNorm)
{
	if (!pNorm)
	{
		return;
	}

	free(pNorm);
}

__global__ void computeRMSNormKernel(nv_bfloat16* output, const nv_bfloat16* input, const nv_bfloat16* weights, float epsilon, int seqLen, int dim)
{
	int idx	   = blockIdx.x * blockDim.x + threadIdx.x;
	int seqIdx = idx / dim;
	int dimIdx = idx % dim;

	if (seqIdx >= seqLen)
		return;

	// Compute sum of squares
	float sumSquare = 0.0f;
	for (int d = 0; d < dim; d++)
	{
		float val = __bfloat162float(input[seqIdx * dim + d]);
		sumSquare += val * val;
	}

	// Compute normalization factor
	float meanSquare = sumSquare / (float)dim;
	float scale		 = 1.0f / sqrtf(meanSquare + epsilon);

	// Apply normalization and weights
	float val	 = __bfloat162float(input[idx]);
	float weight = __bfloat162float(weights[dimIdx]);
	output[idx]	 = __float2bfloat16(val * scale * weight);
}

bool lyRMSNormForward(lyTensor** ppOutput, const lyRMSNorm* pNorm, const lyTensor* pInput)
{
	if (!ppOutput || !pNorm || !pInput || !pInput->data || !pNorm->weights || !pInput->rank)
	{
		return false;
	}

	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput))
	{
		return false;
	}

	int seqLen		  = pInput->shape[0];
	int dim			  = pInput->shape[1];
	int totalElements = seqLen * dim;

	if (!lySetTensorShape(pOutput, pInput->shape, pInput->rank) || !lySetTensorData(pOutput, NULL, totalElements * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	int blockSize = 256;
	int numBlocks = (totalElements + blockSize - 1) / blockSize;

	computeRMSNormKernel<<<numBlocks, blockSize>>>(pOutput->data, pInput->data, pNorm->weights->data, pNorm->epsilon, seqLen, dim);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}