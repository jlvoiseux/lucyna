#include "lyTensorMath.h"
#include "lyTransformerBlock.h"

#include <stdio.h>
#include <stdlib.h>

bool lyCreateTransformerBlock(lyTransformerBlock** ppBlock, const lyModel* pModel, int32_t layerIndex)
{
	if (!ppBlock || !pModel)
	{
		return false;
	}

	lyTransformerBlock* pBlock = (lyTransformerBlock*)malloc(sizeof(lyTransformerBlock));
	if (!pBlock)
	{
		return false;
	}

	pBlock->layerIndex = layerIndex;

	lyTensor* attnNormWeights;
	char	  attnNormName[64];
	snprintf(attnNormName, sizeof(attnNormName), "layers.%d.attention_norm.weight", layerIndex);
	if (!lyGetModelTensor(&attnNormWeights, pModel, attnNormName))
	{
		free(pBlock);
		return false;
	}

	if (!lyCreateRMSNorm(&pBlock->attnNorm, pModel->args.normEps, attnNormWeights))
	{
		free(pBlock);
		return false;
	}

	if (!lyCreateAttention(&pBlock->attention, pModel, layerIndex))
	{
		lyDestroyRMSNorm(pBlock->attnNorm);
		free(pBlock);
		return false;
	}

	lyTensor* ffnNormWeights;
	char	  ffnNormName[64];
	snprintf(ffnNormName, sizeof(ffnNormName), "layers.%d.ffn_norm.weight", layerIndex);
	if (!lyGetModelTensor(&ffnNormWeights, pModel, ffnNormName))
	{
		lyDestroyAttention(pBlock->attention);
		lyDestroyRMSNorm(pBlock->attnNorm);
		free(pBlock);
		return false;
	}

	if (!lyCreateRMSNorm(&pBlock->ffnNorm, pModel->args.normEps, ffnNormWeights))
	{
		lyDestroyAttention(pBlock->attention);
		lyDestroyRMSNorm(pBlock->attnNorm);
		free(pBlock);
		return false;
	}

	if (!lyCreateFeedForward(&pBlock->feedForward, pModel, layerIndex))
	{
		lyDestroyRMSNorm(pBlock->ffnNorm);
		lyDestroyAttention(pBlock->attention);
		lyDestroyRMSNorm(pBlock->attnNorm);
		free(pBlock);
		return false;
	}

	*ppBlock = pBlock;
	return true;
}

void lyDestroyTransformerBlock(lyTransformerBlock* pBlock)
{
	if (!pBlock)
	{
		return;
	}

	lyDestroyFeedForward(pBlock->feedForward);
	lyDestroyRMSNorm(pBlock->ffnNorm);
	lyDestroyAttention(pBlock->attention);
	lyDestroyRMSNorm(pBlock->attnNorm);
	free(pBlock);
}

bool lyTransformerBlockForward(lyTensor** ppOutput, lyTransformerBlock* pBlock, const lyTensor* pInput, int32_t startPos, const lyTensor* pFreqsCis, const lyTensor* pMask)
{
	if (!ppOutput || !pBlock || !pInput || !pFreqsCis)
	{
		return false;
	}

	lyTensor* normalizedInput;
	if (!lyRMSNormForward(&normalizedInput, pBlock->attnNorm, pInput))
	{
		return false;
	}

	lyTensor* attnOutput;
	if (!lyAttentionForward(&attnOutput, pBlock->attention, normalizedInput, startPos, pFreqsCis, pMask))
	{
		lyDestroyTensor(normalizedInput);
		return false;
	}
	lyDestroyTensor(normalizedInput);

	lyTensor* residual;
	if (!lyTensorAdd(&residual, pInput, attnOutput))
	{
		lyDestroyTensor(attnOutput);
		return false;
	}
	lyDestroyTensor(attnOutput);

	lyTensor* normalizedResidual;
	if (!lyRMSNormForward(&normalizedResidual, pBlock->ffnNorm, residual))
	{
		lyDestroyTensor(residual);
		return false;
	}

	lyTensor* ffnOutput;
	if (!lyFeedForwardForward(&ffnOutput, pBlock->feedForward, normalizedResidual))
	{
		lyDestroyTensor(normalizedResidual);
		lyDestroyTensor(residual);
		return false;
	}
	lyDestroyTensor(normalizedResidual);

	if (!lyTensorAdd(ppOutput, residual, ffnOutput))
	{
		lyDestroyTensor(ffnOutput);
		lyDestroyTensor(residual);
		return false;
	}
	lyDestroyTensor(ffnOutput);
	lyDestroyTensor(residual);

	return true;
}