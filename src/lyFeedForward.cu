#include "lyFeedForward.h"
#include "lyModel.h"
#include "lyTensorMath.h"

#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

bool lyCreateFeedForward(lyFeedForward** ppFeedForward, const lyModel* pModel, int32_t layerIndex)
{
	if (!ppFeedForward || !pModel)
	{
		return false;
	}

	lyFeedForward* pFeedForward = (lyFeedForward*)malloc(sizeof(lyFeedForward));
	if (!pFeedForward)
	{
		return false;
	}

	int32_t dim				   = pModel->args.dim;
	pFeedForward->ffnHiddenDim = 4 * dim;
	pFeedForward->ffnHiddenDim = (2 * pFeedForward->ffnHiddenDim) / 3;

	if (pModel->args.ffnDimMultiplier > -1.0f)
	{
		pFeedForward->ffnHiddenDim = (int32_t)(pModel->args.ffnDimMultiplier * pFeedForward->ffnHiddenDim);
	}

	pFeedForward->ffnHiddenDim = pModel->args.multipleOf * ((pFeedForward->ffnHiddenDim + pModel->args.multipleOf - 1) / pModel->args.multipleOf);

	char	tensorName[64];
	int32_t perm[] = {1, 0};

	snprintf(tensorName, sizeof(tensorName), "layers.%d.feed_forward.w1.weight", layerIndex);
	if (!lyGetModelTensor(&pFeedForward->ffnGate, pModel, tensorName))
	{
		lyDestroyFeedForward(pFeedForward);
		return false;
	}
	if (!lyTensorTranspose(&pFeedForward->ffnGate, pFeedForward->ffnGate, perm))
	{
		lyDestroyFeedForward(pFeedForward);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.feed_forward.w2.weight", layerIndex);
	if (!lyGetModelTensor(&pFeedForward->ffnDown, pModel, tensorName))
	{
		lyDestroyFeedForward(pFeedForward);
		return false;
	}
	if (!lyTensorTranspose(&pFeedForward->ffnDown, pFeedForward->ffnDown, perm))
	{
		lyDestroyFeedForward(pFeedForward);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.feed_forward.w3.weight", layerIndex);
	if (!lyGetModelTensor(&pFeedForward->ffnUp, pModel, tensorName))
	{
		lyDestroyFeedForward(pFeedForward);
		return false;
	}
	if (!lyTensorTranspose(&pFeedForward->ffnUp, pFeedForward->ffnUp, perm))
	{
		lyDestroyFeedForward(pFeedForward);
		return false;
	}

	*ppFeedForward = pFeedForward;
	return true;
}

void lyDestroyFeedForward(lyFeedForward* pFeedForward)
{
	if (!pFeedForward)
	{
		return;
	}

	free(pFeedForward);
}

__global__ void siluActivationKernel(nv_bfloat16* output, const nv_bfloat16* input, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size)
		return;

	nv_bfloat16 val = input[idx];
	// SiLU(x) = x * sigmoid(x)
	nv_bfloat16 sigmoid = __hdiv(__float2bfloat16(1.0f), __hadd(__float2bfloat16(1.0f), hexp(__hneg(val))));
	output[idx]			= __hmul(val, sigmoid);
}

bool lyFeedForwardForward(lyTensor** ppOutput, const lyFeedForward* pFeedForward, const lyTensor* pInput)
{
	if (!ppOutput || !pFeedForward || !pInput)
	{
		return false;
	}

	lyTensor* gateResult;
	if (!lyTensorMatMul(&gateResult, pInput, pFeedForward->ffnGate))
	{
		return false;
	}

	int blockSize = 256;
	int numBlocks = (gateResult->shape[0] * gateResult->shape[1] + blockSize - 1) / blockSize;

	siluActivationKernel<<<numBlocks, blockSize>>>(gateResult->data, gateResult->data, gateResult->shape[0] * gateResult->shape[1]);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(gateResult);
		return false;
	}

	lyTensor* upResult;
	if (!lyTensorMatMul(&upResult, pInput, pFeedForward->ffnUp))
	{
		lyDestroyTensor(gateResult);
		return false;
	}

	lyTensor* elementwiseProduct;
	if (!lyTensorElementwiseMul(&elementwiseProduct, gateResult, upResult))
	{
		lyDestroyTensor(gateResult);
		lyDestroyTensor(upResult);
		return false;
	}

	lyDestroyTensor(gateResult);
	lyDestroyTensor(upResult);

	if (!lyTensorMatMul(ppOutput, elementwiseProduct, pFeedForward->ffnDown))
	{
		lyDestroyTensor(elementwiseProduct);
		return false;
	}

	lyDestroyTensor(elementwiseProduct);
	return true;
}