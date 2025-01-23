#pragma once

#include "lyAttention.h"
#include "lyFeedForward.h"
#include "lyModel.h"
#include "lyRMSNorm.h"

typedef struct lyTransformerBlock
{
	int32_t layerIndex;

	lyRMSNorm* attnNorm;
	lyRMSNorm* ffnNorm;

	lyAttention*   attention;
	lyFeedForward* feedForward;
} lyLlamaTransformerBlock;

bool lyCreateTransformerBlock(lyTransformerBlock** ppBlock, const lyModel* pModel, int32_t layerIndex);
void lyDestroyTransformerBlock(lyTransformerBlock* pBlock);

bool lyTransformerBlockForward(lyTensor** ppOutput, lyTransformerBlock* pBlock, lyTensor* pInput, int32_t startPos, lyTensor* pFreqsCis, lyTensor* pMask);