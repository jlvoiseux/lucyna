#pragma once

#include "lyRMSNorm.h"
#include "lyTensor.h"
#include "lyTransformerBlock.h"

typedef struct lyTransformer
{
	int32_t dim;
	int32_t vocabSize;
	int32_t nLayers;

	lyTensor*			 tokEmbeddings;
	lyTransformerBlock** blocks;
	lyRMSNorm*			 norm;
	lyTensor*			 output;

	lyTensor* freqsCis;
} lyTransformer;

void lyTransformerCreate(lyTransformer** ppTransformer, const lyModel* pModel);
void lyTransformerDestroy(lyTransformer* pTransformer);

void lyTransformerForward(lyTensor** ppOutput, lyTransformer* pTransformer, lyTensor* pTokens, int32_t startPos);