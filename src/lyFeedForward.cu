#include "lyFeedForward.h"
#include "lyModel.h"
#include "lyTensorMath.h"

#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

void lyFeedForwardCreate(lyFeedForward** ppFeedForward, const lyModel* pModel, int32_t layerIndex)
{
	lyFeedForward* pFeedForward = (lyFeedForward*)malloc(sizeof(lyFeedForward));

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
	lyTensor* modelGate;
	lyModelGetTensor(&modelGate, pModel, tensorName);
	lyTensorTranspose(&pFeedForward->ffnGate, modelGate, perm);

	snprintf(tensorName, sizeof(tensorName), "layers.%d.feed_forward.w2.weight", layerIndex);
	lyTensor* modelDown;
	lyModelGetTensor(&modelDown, pModel, tensorName);
	lyTensorTranspose(&pFeedForward->ffnDown, modelDown, perm);

	snprintf(tensorName, sizeof(tensorName), "layers.%d.feed_forward.w3.weight", layerIndex);
	lyTensor* modelUp;
	lyModelGetTensor(&modelUp, pModel, tensorName);
	lyTensorTranspose(&pFeedForward->ffnUp, modelUp, perm);

	*ppFeedForward = pFeedForward;
}

void lyFeedForwardDestroy(lyFeedForward* pFeedForward)
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

void lyFeedForwardForward(lyTensor** ppOutput, const lyFeedForward* pFeedForward, lyTensor* pInput)
{
	lyTensorPrint(pInput);

	lyTensor* gateResult;
	lyTensorMatMul(&gateResult, pInput, pFeedForward->ffnGate);
	lyTensorPrint(gateResult);

	int totalElements = gateResult->shape[0] * gateResult->shape[1];
	for (int i = 0; i < totalElements; i++)
	{
		float val			= __bfloat162float(gateResult->data[i]);
		gateResult->data[i] = __float2bfloat16_rz(silu(val));
	}
	lyTensorPrint(gateResult);

	lyTensor* upResult;
	lyTensorMatMul(&upResult, pInput, pFeedForward->ffnUp);
	lyTensorPrint(upResult);

	lyTensor* elementwiseProduct;
	lyTensorElementwiseMul(&elementwiseProduct, gateResult, upResult);
	lyTensorDestroy(gateResult);
	lyTensorDestroy(upResult);
	lyTensorPrint(elementwiseProduct);

	lyTensorMatMul(ppOutput, elementwiseProduct, pFeedForward->ffnDown);
	lyTensorDestroy(elementwiseProduct);
}