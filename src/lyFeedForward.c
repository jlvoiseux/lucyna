#include "lyFeedForward.h"

#include "lyModel.h"
#include "lyTensorMath.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void lyFeedForwardCreate(lyFeedForward** ppFeedForward, const lyModel* pModel, int32_t layerIndex, lyOpenCLContext* pContext)
{
	lyFeedForward* pFeedForward = (lyFeedForward*)malloc(sizeof(lyFeedForward));

	int32_t dim					= pModel->args.dim;
	pFeedForward->ffnHiddenDim	= 4 * dim;
	pFeedForward->ffnHiddenDim	= (2 * pFeedForward->ffnHiddenDim) / 3;
	pFeedForward->openCLContext = pContext;

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
	lyTensorTranspose(&pFeedForward->ffnGate, modelGate, perm, pContext);

	snprintf(tensorName, sizeof(tensorName), "layers.%d.feed_forward.w2.weight", layerIndex);
	lyTensor* modelDown;
	lyModelGetTensor(&modelDown, pModel, tensorName);
	lyTensorTranspose(&pFeedForward->ffnDown, modelDown, perm, pContext);

	snprintf(tensorName, sizeof(tensorName), "layers.%d.feed_forward.w3.weight", layerIndex);
	lyTensor* modelUp;
	lyModelGetTensor(&modelUp, pModel, tensorName);
	lyTensorTranspose(&pFeedForward->ffnUp, modelUp, perm, pContext);

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
	lyTensorMatMul(&gateResult, pInput, pFeedForward->ffnGate, pFeedForward->openCLContext);
	lyTensorPrint(gateResult);

	int totalElements = gateResult->shape[0] * gateResult->shape[1];
	for (int i = 0; i < totalElements; i++)
	{
		float val			= lyBfloat16ToFloat32(gateResult->data[i]);
		gateResult->data[i] = lyFloat32ToBfloat16(silu(val));
	}
	lyTensorPrint(gateResult);

	lyTensor* upResult;
	lyTensorMatMul(&upResult, pInput, pFeedForward->ffnUp, pFeedForward->openCLContext);
	lyTensorPrint(upResult);

	lyTensor* elementwiseProduct;
	lyTensorElementwiseMul(&elementwiseProduct, gateResult, upResult, pFeedForward->openCLContext);
	lyTensorDestroy(gateResult);
	lyTensorDestroy(upResult);
	lyTensorPrint(elementwiseProduct);

	lyTensorMatMul(ppOutput, elementwiseProduct, pFeedForward->ffnDown, pFeedForward->openCLContext);
	lyTensorDestroy(elementwiseProduct);
}