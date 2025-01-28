#include "lyAttention.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"

#include <assert.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

void lyAttentionCreate(lyAttention** ppAttention, const lyModel* pModel, int32_t layerIndex)
{
	lyAttention* pAttention = (lyAttention*)malloc(sizeof(lyAttention));

	pAttention->layerIndex = layerIndex;
	pAttention->nHeads	   = pModel->args.nHeads;
	pAttention->nKVHeads   = pModel->args.nKVHeads;
	pAttention->nRep	   = pModel->args.nRep;
	pAttention->headDim	   = pModel->args.headDim;

	char	tensorName[64];
	int32_t perm[] = {1, 0};

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wq.weight", layerIndex);
	lyTensor* modelWQ;
	lyModelGetTensor(&modelWQ, pModel, tensorName);
	lyTensorTranspose(&pAttention->attnWQ, modelWQ, perm);

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wk.weight", layerIndex);
	lyTensor* modelWK;
	lyModelGetTensor(&modelWK, pModel, tensorName);
	lyTensorTranspose(&pAttention->attnWK, modelWK, perm);

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wv.weight", layerIndex);
	lyTensor* modelWV;
	lyModelGetTensor(&modelWV, pModel, tensorName);
	lyTensorTranspose(&pAttention->attnWV, modelWV, perm);

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wo.weight", layerIndex);
	lyTensor* modelWO;
	lyModelGetTensor(&modelWO, pModel, tensorName);
	lyTensorTranspose(&pAttention->attnWO, modelWO, perm);

	int32_t	  cacheShape[] = {pModel->args.maxSequenceLength, pModel->args.nKVHeads, pModel->args.headDim};
	lyTensor* pCacheK;
	lyTensor* pCacheV;
	lyTensorCreate(&pCacheK, cacheShape, 3, NULL, NULL);
	lyTensorCreate(&pCacheV, cacheShape, 3, NULL, NULL);
	pAttention->cacheK = pCacheK;
	pAttention->cacheV = pCacheV;
	*ppAttention	   = pAttention;
}

void lyAttentionDestroy(lyAttention* pAttention)
{
	if (!pAttention)
	{
		return;
	}

	lyTensorDestroy(pAttention->cacheK);
	lyTensorDestroy(pAttention->cacheV);
	free(pAttention);
}

void lyRepeatKV(lyTensor** ppOutput, lyTensor* pInput, int32_t nRep)
{
	if (nRep == 1)
	{
		lyTensor* pOutput;
		lyTensorCreate(&pOutput, pInput->shape, pInput->rank, pInput->data, NULL);
		*ppOutput = pOutput;
		return;
	}

	int32_t seqLen	 = pInput->shape[0];
	int32_t nKVHeads = pInput->shape[1];
	int32_t headDim	 = pInput->shape[2];

	lyTensor* pIntermediate;
	int32_t	  intermediateShape[] = {seqLen, nKVHeads, nRep, headDim};
	lyTensorCreate(&pIntermediate, intermediateShape, 4, NULL, NULL);

	for (int32_t i = 0; i < seqLen; i++)
	{
		for (int32_t j = 0; j < nKVHeads; j++)
		{
			size_t srcOffset = (i * nKVHeads + j) * headDim;
			for (int32_t rep = 0; rep < nRep; rep++)
			{
				size_t dstOffset = ((i * nKVHeads + j) * nRep + rep) * headDim;
				memcpy(pIntermediate->data + dstOffset, pInput->data + srcOffset, headDim * sizeof(nv_bfloat16));
			}
		}
	}

	lyTensor* pOutput;
	int32_t	  outputShape[] = {seqLen, nKVHeads * nRep, headDim};
	lyTensorCreate(&pOutput, outputShape, 3, pIntermediate->data, NULL);

	lyTensorDestroy(pIntermediate);
	*ppOutput = pOutput;
}

void lyUpdateKVCache(lyAttention* pAttention, lyTensor* pK, lyTensor* pV, int32_t startPos)
{
	int32_t seqLen = pK->shape[0];

	int32_t elementsPerPos = pAttention->nKVHeads * pAttention->headDim;
	size_t	offsetElements = startPos * elementsPerPos;
	size_t	copyElements   = seqLen * elementsPerPos;
	size_t	copyBytes	   = copyElements * sizeof(nv_bfloat16);

	nv_bfloat16* kDst = pAttention->cacheK->data + offsetElements;
	nv_bfloat16* vDst = pAttention->cacheV->data + offsetElements;

	memcpy(kDst, pK->data, copyBytes);
	memcpy(vDst, pV->data, copyBytes);

	lyTensorPrint(pAttention->cacheK);
	lyTensorPrint(pAttention->cacheV);
}

void lyAttentionForward(lyTensor** ppOutput, lyAttention* pAttention, lyTensor* pInput, int32_t startPos, lyTensorDouble* pFreqsCis, lyTensor* pMask)
{
	lyTensorPrint(pInput);

	lyTensor *xq, *xk, *xv;
	lyTensorMatMul(&xq, pInput, pAttention->attnWQ);
	lyTensorMatMul(&xk, pInput, pAttention->attnWK);
	lyTensorMatMul(&xv, pInput, pAttention->attnWV);
	lyTensorPrint(xq);
	lyTensorPrint(xk);
	lyTensorPrint(xv);

	int32_t seqLen	  = pInput->shape[0];
	int32_t qShape[]  = {seqLen, pAttention->nHeads, pAttention->headDim};
	int32_t kvShape[] = {seqLen, pAttention->nKVHeads, pAttention->headDim};
	lyTensorReshape(xq, qShape, 3);
	lyTensorReshape(xk, kvShape, 3);
	lyTensorReshape(xv, kvShape, 3);
	lyTensorPrint(xq);
	lyTensorPrint(xk);
	lyTensorPrint(xv);

	lyTensor *rotatedQ, *rotatedK;
	lyRopeApplyEmbeddings(&rotatedQ, &rotatedK, xq, xk, pFreqsCis);
	lyTensorDestroy(xq);
	lyTensorDestroy(xk);
	lyTensorPrint(rotatedQ);
	lyTensorPrint(rotatedK);

	lyUpdateKVCache(pAttention, rotatedK, xv, startPos);

	lyTensor *keys, *values;
	lyTensorSlice(&keys, pAttention->cacheK, 0, startPos + seqLen);
	lyTensorSlice(&values, pAttention->cacheV, 0, startPos + seqLen);
	lyTensorPrint(keys);
	lyTensorPrint(values);

	lyTensor *repeatedKeys, *repeatedValues;
	lyRepeatKV(&repeatedKeys, keys, pAttention->nRep);
	lyRepeatKV(&repeatedValues, values, pAttention->nRep);
	lyTensorPrint(repeatedKeys);
	lyTensorPrint(repeatedValues);

	lyTensorDestroy(keys);
	lyTensorDestroy(values);
	lyTensorDestroy(rotatedK);
	lyTensorDestroy(xv);

	lyTensor *transposedQ, *transposedK, *transposedV, *reTransposedK;
	int32_t	  perm1[] = {1, 0, 2};
	int32_t	  perm2[] = {0, 2, 1};

	lyTensorTranspose(&transposedQ, rotatedQ, perm1);
	lyTensorTranspose(&transposedK, repeatedKeys, perm1);
	lyTensorTranspose(&reTransposedK, transposedK, perm2);
	lyTensorTranspose(&transposedV, repeatedValues, perm1);
	lyTensorDestroy(rotatedQ);
	lyTensorDestroy(repeatedKeys);
	lyTensorDestroy(repeatedValues);
	lyTensorDestroy(transposedK);
	lyTensorPrint(transposedQ);
	lyTensorPrint(reTransposedK);
	lyTensorPrint(transposedV);

	lyTensor* scores;
	lyTensorMatMul(&scores, transposedQ, reTransposedK);
	lyTensorPrint(transposedQ);
	lyTensorPrint(reTransposedK);
	lyTensorPrint(scores);

	nv_bfloat16 scaleFactor = __float2bfloat16_rz((float)(1.0 / sqrt((double)pAttention->headDim)));
	lyTensor*	scaledScores;
	lyTensorScaleAndAdd(&scaledScores, scores, pMask, scaleFactor, __float2bfloat16_rz(1.f));
	lyTensorDestroy(scores);
	lyTensorPrint(scaledScores);

	lyTensor* softmaxScores;
	lyTensorSoftmax(&softmaxScores, scaledScores);
	lyTensorDestroy(scaledScores);
	lyTensorPrint(softmaxScores);

	lyTensor* attnOut;
	lyTensorMatMul(&attnOut, softmaxScores, transposedV);
	lyTensorDestroy(softmaxScores);
	lyTensorDestroy(transposedV);
	lyTensorPrint(attnOut);

	lyTensor* reTransposed;
	int32_t	  transposePerm[] = {1, 0, 2};
	lyTensorTranspose(&reTransposed, attnOut, transposePerm);
	lyTensorDestroy(attnOut);
	lyTensorPrint(reTransposed);

	int32_t flatShape[] = {seqLen, pAttention->nHeads * pAttention->headDim};
	lyTensorReshape(reTransposed, flatShape, 2);
	lyTensorPrint(reTransposed);

	lyTensorMatMul(ppOutput, reTransposed, pAttention->attnWO);
	lyTensorPrint(*ppOutput);

	lyTensorDestroy(reTransposed);
}