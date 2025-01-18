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

bool lyCreateTransformer(lyTransformer** ppTransformer, const lyModel* pModel);
void lyDestroyTransformer(lyTransformer* pTransformer);

bool lyTransformerForward(lyTensor** ppOutput, lyTransformer* pTransformer, const lyTensor* pTokens, int32_t startPos);