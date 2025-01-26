#pragma once

#include "lyModel.h"
#include "lyTensor.h"

typedef struct lyAttention
{
	int32_t layerIndex;
	int32_t nHeads;
	int32_t nKVHeads;
	int32_t nRep;
	int32_t headDim;

	lyTensor* attnWQ;
	lyTensor* attnWK;
	lyTensor* attnWV;
	lyTensor* attnWO;

	lyTensor* cacheK;
	lyTensor* cacheV;
} lyAttention;

void lyAttentionCreate(lyAttention** ppAttention, const lyModel* pModel, int32_t layerIndex);
void lyAttentionDestroy(lyAttention* pAttention);
void lyAttentionForward(lyTensor** ppOutput, lyAttention* pAttention, lyTensor* pInput, int32_t startPos, lyTensor* pFreqsCis, lyTensor* pMask);