#include "lyModel.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"
#include "lyTransformer.h"

#include <stdio.h>
#include <stdlib.h>

void lyTransformerCreate(lyTransformer** ppTransformer, const lyModel* pModel)
{
	lyTransformer* pTransformer = (lyTransformer*)malloc(sizeof(lyTransformer));
	pTransformer->dim			= pModel->args.dim;
	pTransformer->vocabSize		= pModel->args.vocabSize;
	pTransformer->nLayers		= pModel->args.nLayers;

	lyModelGetTensor(&pTransformer->tokEmbeddings, pModel, "tok_embeddings.weight");
	pTransformer->blocks = (lyTransformerBlock**)malloc(sizeof(lyTransformerBlock*) * pTransformer->nLayers);
	for (int32_t i = 0; i < pTransformer->nLayers; i++)
		lyTransformerBlockCreate(&pTransformer->blocks[i], pModel, i);

	lyTensor* normWeights;
	lyModelGetTensor(&normWeights, pModel, "norm.weight");
	lyRMSNormCreate(&pTransformer->norm, pModel->args.normEps, normWeights);

	int32_t	  perm[] = {1, 0};
	lyTensor* tempOutput;
	lyModelGetTensor(&tempOutput, pModel, "output.weight");
	lyTensorTranspose(&pTransformer->output, tempOutput, perm);
	lyRopePrecomputeFreqsCis(&pTransformer->freqsCis, pTransformer->dim / pModel->args.nHeads, pModel->args.maxSequenceLength * 2, pModel->args.ropeTheta);
	*ppTransformer = pTransformer;
}

void lyTransformerDestroy(lyTransformer* pTransformer)
{
	if (pTransformer->blocks)
	{
		for (int32_t i = 0; i < pTransformer->nLayers; i++)
		{
			lyTransformerBlockDestroy(pTransformer->blocks[i]);
		}
		free(pTransformer->blocks);
	}

	lyRMSNormDestroy(pTransformer->norm);
	lyTensorDestroy(pTransformer->freqsCis);
	free(pTransformer);
}

void lyTransformerForward(lyTensor** ppOutput, lyTransformer* pTransformer, lyTensor* pTokens, int32_t startPos)
{
	lyTensor* h;
	lyTensorEmbedding(&h, pTokens, pTransformer->tokEmbeddings);
	lyTensorPrint(h);

	int32_t	  seqLen = pTokens->shape[0];
	lyTensor* freqsCis;
	lyTensorSlice(&freqsCis, pTransformer->freqsCis, startPos, startPos + seqLen);
	lyTensorPrint(freqsCis);

	lyTensor* mask = NULL;
	if (seqLen > 1)
	{
		int32_t shape[] = {seqLen, seqLen + startPos};
		lyTensorCreate(&mask, shape, 2, NULL, NULL);
		lyTensorMakeTriangularMask(mask);
		lyTensorPrint(mask);
	}

	lyTensor* currentTensor = h;
	for (int32_t i = 0; i < pTransformer->nLayers; i++)
	{
		lyTensor* blockOut;
		lyTransformerBlockForward(&blockOut, pTransformer->blocks[i], currentTensor, startPos, freqsCis, mask);
		if (currentTensor != h)
			lyTensorDestroy(currentTensor);
		currentTensor = blockOut;
		lyTensorPrint(currentTensor);
	}

	if (mask)
		lyTensorDestroy(mask);
	lyTensorDestroy(freqsCis);

	lyTensor* normalized;
	lyRMSNormForward(&normalized, pTransformer->norm, currentTensor);
	lyTensorDestroy(currentTensor);
	lyTensorPrint(normalized);

	lyTensorMatMul(ppOutput, normalized, pTransformer->output);
	lyTensorDestroy(normalized);
}