#include "lyModel.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"
#include "lyTransformer.h"

#include <stdlib.h>

bool lyCreateTransformer(lyTransformer** ppTransformer, const lyModel* pModel)
{
	if (!ppTransformer || !pModel)
	{
		return false;
	}

	lyTransformer* pTransformer = (lyTransformer*)malloc(sizeof(lyTransformer));
	if (!pTransformer)
	{
		return false;
	}

	pTransformer->dim		= pModel->args.dim;
	pTransformer->vocabSize = pModel->args.vocabSize;
	pTransformer->nLayers	= pModel->args.nLayers;

	lyTensor* pEmbeddings;
	if (!lyGetModelTensor(&pEmbeddings, pModel, "tok_embeddings.weight"))
	{
		lyDestroyTransformer(pTransformer);
		return false;
	}
	pTransformer->tokEmbeddings = pEmbeddings;

	pTransformer->blocks = (lyTransformerBlock**)malloc(sizeof(lyTransformerBlock*) * pTransformer->nLayers);
	if (!pTransformer->blocks)
	{
		lyDestroyTransformer(pTransformer);
		return false;
	}

	for (int32_t i = 0; i < pTransformer->nLayers; i++)
	{
		if (!lyCreateTransformerBlock(&pTransformer->blocks[i], pModel, i))
		{
			lyDestroyTransformer(pTransformer);
			return false;
		}
	}

	lyTensor* normWeights;
	if (!lyGetModelTensor(&normWeights, pModel, "norm.weight"))
	{
		lyDestroyTransformer(pTransformer);
		return false;
	}
	if (!lyCreateRMSNorm(&pTransformer->norm, pModel->args.normEps, normWeights))
	{
		lyDestroyTensor(normWeights);
		lyDestroyTransformer(pTransformer);
		return false;
	}

	int32_t	  perm[] = {1, 0};
	lyTensor* output;
	lyTensor* tempOutput;
	if (!lyGetModelTensor(&tempOutput, pModel, "output.weight"))
	{
		lyDestroyTransformer(pTransformer);
		return false;
	}
	if (!lyTensorTranspose(&output, tempOutput, perm))
	{
		lyDestroyTensor(tempOutput);
		lyDestroyTransformer(pTransformer);
		return false;
	}
	pTransformer->output = output;

	if (!precomputeFreqsCis(&pTransformer->freqsCis, pTransformer->dim / pModel->args.nHeads, pModel->args.maxSequenceLength * 2, pModel->args.ropeTheta))
	{
		lyDestroyTransformer(pTransformer);
		return false;
	}

	*ppTransformer = pTransformer;
	return true;
}

void lyDestroyTransformer(lyTransformer* pTransformer)
{
	if (!pTransformer)
	{
		return;
	}

	lyDestroyTensor(pTransformer->tokEmbeddings);

	if (pTransformer->blocks)
	{
		for (int32_t i = 0; i < pTransformer->nLayers; i++)
		{
			lyDestroyTransformerBlock(pTransformer->blocks[i]);
		}
		free(pTransformer->blocks);
	}

	lyDestroyRMSNorm(pTransformer->norm);
	lyDestroyTensor(pTransformer->output);
	lyDestroyTensor(pTransformer->freqsCis);
	free(pTransformer);
}

bool lyTransformerForward(lyTensor** ppOutput, lyTransformer* pTransformer, lyTensor* pTokens, int32_t startPos)
{
	if (!ppOutput || !pTransformer || !pTokens || startPos < 0)
	{
		return false;
	}

	lyTensor* h;
	if (!lyTensorEmbedding(&h, pTokens, pTransformer->tokEmbeddings))
	{
		return false;
	}

	int32_t	  seqLen = pTokens->shape[0];
	lyTensor* freqsCis;
	lyTensorSlice(&freqsCis, pTransformer->freqsCis, startPos, startPos + seqLen);

	lyTensor* mask = NULL;
	if (seqLen > 1)
	{
		int32_t shape[] = {seqLen, seqLen};
		lyCreateTensor(&mask, shape, 2, NULL, NULL);
	}

	lyTensor* currentTensor = h;
	for (int32_t i = 0; i < pTransformer->nLayers; i++)
	{
		lyTensor* blockOut;
		if (!lyTransformerBlockForward(&blockOut, pTransformer->blocks[i], currentTensor, startPos, freqsCis, mask))
		{
			if (mask)
				lyDestroyTensor(mask);
			lyDestroyTensor(freqsCis);
			lyDestroyTensor(currentTensor);
			return false;
		}

		if (currentTensor != h)
		{
			lyDestroyTensor(currentTensor);
		}
		currentTensor = blockOut;
	}

	if (mask)
		lyDestroyTensor(mask);
	lyDestroyTensor(freqsCis);

	lyTensor* normalized;
	if (!lyRMSNormForward(&normalized, pTransformer->norm, currentTensor))
	{
		lyDestroyTensor(currentTensor);
		return false;
	}
	lyDestroyTensor(currentTensor);

	if (!lyTensorMatMul(ppOutput, normalized, pTransformer->output))
	{
		lyDestroyTensor(normalized);
		return false;
	}
	lyDestroyTensor(normalized);

	return true;
}