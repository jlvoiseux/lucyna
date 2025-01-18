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
	int32_t	  embeddingsShape[] = {pTransformer->vocabSize, pTransformer->dim};
	if (!lyGetModelTensor(&pEmbeddings, pModel, "tok_embeddings.weight") || !lySetTensorShape(pEmbeddings, embeddingsShape, 2))
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
		lyDestroyTransformer(pTransformer);
		return false;
	}

	lyTensor* output;
	if (!lyGetModelTensor(&output, pModel, "output.weight") || !lySetTensorShape(output, embeddingsShape, 2))
	{
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

bool lyTransformerForward(lyTensor** ppOutput, lyTransformer* pTransformer, const lyTensor* pTokens, int32_t startPos)
{
	if (!ppOutput || !pTransformer || !pTokens || startPos < 0)
	{
		return false;
	}

	lyTensor* h;
	if (!lyTensorMatMul(&h, pTokens, pTransformer->tokEmbeddings))
	{
		return false;
	}

	int32_t	  seqLen = pTokens->shape[0];
	lyTensor* freqsCis;
	if (!lyTensorSlice(&freqsCis, pTransformer->freqsCis, startPos, startPos + seqLen))
	{
		lyDestroyTensor(h);
		return false;
	}

	lyTensor* mask = NULL;
	if (seqLen > 1)
	{
		int32_t shape[] = {seqLen, seqLen};
		if (!lyCreateTensor(&mask))
		{
			lyDestroyTensor(freqsCis);
			lyDestroyTensor(h);
			return false;
		}
		if (!lySetTensorShape(mask, shape, 2))
		{
			lyDestroyTensor(mask);
			lyDestroyTensor(freqsCis);
			lyDestroyTensor(h);
			return false;
		}

		if (!lyTensorToGPU(mask))
		{
			lyDestroyTensor(mask);
			lyDestroyTensor(freqsCis);
			lyDestroyTensor(h);
			return false;
		}

		if (!lyTensorMakeTriangularMask(mask))
		{
			lyDestroyTensor(mask);
			lyDestroyTensor(freqsCis);
			lyDestroyTensor(h);
			return false;
		}
	}

	for (int32_t i = 0; i < pTransformer->nLayers; i++)
	{
		lyTensor* blockOut;
		if (!lyTransformerBlockForward(&blockOut, pTransformer->blocks[i], h, startPos, freqsCis, mask))
		{
			if (mask)
				lyDestroyTensor(mask);
			lyDestroyTensor(freqsCis);
			lyDestroyTensor(h);
			return false;
		}
		lyDestroyTensor(h);
		h = blockOut;
	}

	if (mask)
		lyDestroyTensor(mask);
	lyDestroyTensor(freqsCis);

	lyTensor* normalized;
	if (!lyRMSNormForward(&normalized, pTransformer->norm, h))
	{
		lyDestroyTensor(h);
		return false;
	}
	lyDestroyTensor(h);

	if (!lyTensorMatMul(ppOutput, normalized, pTransformer->output))
	{
		lyDestroyTensor(normalized);
		return false;
	}
	lyDestroyTensor(normalized);

	return true;
}