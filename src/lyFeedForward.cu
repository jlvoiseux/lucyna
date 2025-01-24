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
	lyTensor* tempGate;
	if (!lyGetModelTensor(&tempGate, pModel, tensorName))
	{
		lyDestroyFeedForward(pFeedForward);
		return false;
	}
	if (!lyTensorTranspose(&pFeedForward->ffnGate, tempGate, perm))
	{
		lyDestroyTensor(tempGate);
		lyDestroyFeedForward(pFeedForward);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.feed_forward.w2.weight", layerIndex);
	lyTensor* tempDown;
	if (!lyGetModelTensor(&tempDown, pModel, tensorName))
	{
		lyDestroyFeedForward(pFeedForward);
		return false;
	}
	if (!lyTensorTranspose(&pFeedForward->ffnDown, tempDown, perm))
	{
		lyDestroyTensor(tempDown);
		lyDestroyFeedForward(pFeedForward);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.feed_forward.w3.weight", layerIndex);
	lyTensor* tempUp;
	if (!lyGetModelTensor(&tempUp, pModel, tensorName))
	{
		lyDestroyFeedForward(pFeedForward);
		return false;
	}
	if (!lyTensorTranspose(&pFeedForward->ffnUp, tempUp, perm))
	{
		lyDestroyTensor(tempUp);
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

static float silu(float x)
{
	return x / (1.0f + expf(-x));
}

bool lyFeedForwardForward(lyTensor** ppOutput, const lyFeedForward* pFeedForward, lyTensor* pInput)
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

	int totalElements = gateResult->shape[0] * gateResult->shape[1];
	for (int i = 0; i < totalElements; i++)
	{
		float val			= __bfloat162float(gateResult->data[i]);
		gateResult->data[i] = __float2bfloat16(silu(val));
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