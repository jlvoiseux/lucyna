#include "lyTensorMath.h"
#include "lyTransformerBlock.h"

#include <stdio.h>
#include <stdlib.h>

void lyTransformerBlockCreate(lyTransformerBlock** ppBlock, const lyModel* pModel, int32_t layerIndex)
{
	lyTransformerBlock* pBlock = (lyTransformerBlock*)malloc(sizeof(lyTransformerBlock));
	pBlock->layerIndex		   = layerIndex;

	lyTensor* attnNormWeights;
	char	  attnNormName[64];
	snprintf(attnNormName, sizeof(attnNormName), "layers.%d.attention_norm.weight", layerIndex);
	lyModelGetTensor(&attnNormWeights, pModel, attnNormName);
	lyRMSNormCreate(&pBlock->attnNorm, pModel->args.normEps, attnNormWeights);
	lyTensorPrint(attnNormWeights);
	lyAttentionCreate(&pBlock->attention, pModel, layerIndex);

	lyTensor* ffnNormWeights;
	char	  ffnNormName[64];
	snprintf(ffnNormName, sizeof(ffnNormName), "layers.%d.ffn_norm.weight", layerIndex);
	lyModelGetTensor(&ffnNormWeights, pModel, ffnNormName);
	lyRMSNormCreate(&pBlock->ffnNorm, pModel->args.normEps, ffnNormWeights);
	lyFeedForwardCreate(&pBlock->feedForward, pModel, layerIndex);

	*ppBlock = pBlock;
}

void lyTransformerBlockDestroy(lyTransformerBlock* pBlock)
{
	if (!pBlock)
	{
		return;
	}

	lyFeedForwardDestroy(pBlock->feedForward);
	lyRMSNormDestroy(pBlock->ffnNorm);
	lyAttentionDestroy(pBlock->attention);
	lyRMSNormDestroy(pBlock->attnNorm);
	free(pBlock);
}

void lyTransformerBlockForward(lyTensor** ppOutput, lyTransformerBlock* pBlock, lyTensor* pInput, int32_t startPos, lyTensorDouble* pFreqsCis, lyTensor* pMask)
{
	lyTensorPrint(pInput);

	lyTensor* pNormalizedInput;
	lyRMSNormForward(&pNormalizedInput, pBlock->attnNorm, pInput);
	lyTensorPrint(pNormalizedInput);

	lyTensor* pAttnOutput;
	lyAttentionForward(&pAttnOutput, pBlock->attention, pNormalizedInput, startPos, pFreqsCis, pMask);
	lyTensorDestroy(pNormalizedInput);
	lyTensorPrint(pAttnOutput);

	lyTensor* pResidual;
	lyTensorScaleAndAdd(&pResidual, pInput, pAttnOutput, __float2bfloat16_rz(1.f), __float2bfloat16_rz(1.f));
	lyTensorDestroy(pAttnOutput);
	lyTensorPrint(pResidual);

	lyTensor* normalizedResidual;
	lyRMSNormForward(&normalizedResidual, pBlock->ffnNorm, pResidual);
	lyTensorPrint(normalizedResidual);

	lyTensor* ffnOutput;
	lyFeedForwardForward(&ffnOutput, pBlock->feedForward, normalizedResidual);
	lyTensorDestroy(normalizedResidual);
	lyTensorPrint(ffnOutput);

	lyTensorScaleAndAdd(ppOutput, pResidual, ffnOutput, __float2bfloat16_rz(1.f), __float2bfloat16_rz(1.f));
	lyTensorDestroy(ffnOutput);
	lyTensorDestroy(pResidual);
}