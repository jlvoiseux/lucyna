#include "lyAttention.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"

#include <assert.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

bool lyCreateAttention(lyAttention** ppAttention, const lyModel* pModel, int32_t layerIndex)
{
	if (!ppAttention || !pModel)
	{
		return false;
	}

	lyAttention* pAttention = (lyAttention*)malloc(sizeof(lyAttention));
	if (!pAttention)
	{
		return false;
	}

	pAttention->layerIndex = layerIndex;
	pAttention->nHeads	   = pModel->args.nHeads;
	pAttention->nKVHeads   = pModel->args.nKVHeads;
	pAttention->nRep	   = pModel->args.nRep;
	pAttention->headDim	   = pModel->args.headDim;

	char	tensorName[64];
	int32_t perm[] = {1, 0};

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wq.weight", layerIndex);
	lyTensor* tempWQ;
	if (!lyGetModelTensor(&tempWQ, pModel, tensorName))
	{
		free(pAttention);
		return false;
	}
	if (!lyTensorTranspose(&pAttention->attnWQ, tempWQ, perm))
	{
		lyDestroyTensor(tempWQ);
		free(pAttention);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wk.weight", layerIndex);
	lyTensor* tempWK;
	if (!lyGetModelTensor(&tempWK, pModel, tensorName))
	{
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lyTensorTranspose(&pAttention->attnWK, tempWK, perm))
	{
		lyDestroyTensor(tempWK);
		lyDestroyAttention(pAttention);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wv.weight", layerIndex);
	lyTensor* tempWV;
	if (!lyGetModelTensor(&tempWV, pModel, tensorName))
	{
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lyTensorTranspose(&pAttention->attnWV, tempWV, perm))
	{
		lyDestroyTensor(tempWV);
		lyDestroyAttention(pAttention);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wo.weight", layerIndex);
	lyTensor* tempWO;
	if (!lyGetModelTensor(&tempWO, pModel, tensorName))
	{
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lyTensorTranspose(&pAttention->attnWO, tempWO, perm))
	{
		lyDestroyTensor(tempWO);
		lyDestroyAttention(pAttention);
		return false;
	}

	int32_t cacheShape[] = {pModel->args.maxSequenceLength, pModel->args.nKVHeads, pModel->args.headDim};

	lyTensor* pCacheK;
	lyCreateTensor(&pCacheK, cacheShape, 3, NULL, NULL);

	lyTensor* pCacheV;
	lyCreateTensor(&pCacheV, cacheShape, 3, NULL, NULL);

	pAttention->cacheK = pCacheK;
	pAttention->cacheV = pCacheV;

	*ppAttention = pAttention;

	return true;
}

void lyDestroyAttention(lyAttention* pAttention)
{
	if (!pAttention)
	{
		return;
	}

	lyDestroyTensor(pAttention->cacheK);
	lyDestroyTensor(pAttention->cacheV);
	free(pAttention);
}

bool lyRepeatKV(lyTensor** ppOutput, lyTensor* pInput, int32_t nRep)
{
	if (!ppOutput || !pInput || nRep <= 0)
	{
		return false;
	}

	if (nRep == 1)
	{
		lyTensor* pOutput;
		lyCreateTensor(&pOutput, pInput->shape, pInput->rank, pInput->data, NULL);
		*ppOutput = pOutput;
		return true;
	}

	if (pInput->rank != 3)
	{
		return false;
	}

	int32_t seqLen	 = pInput->shape[0];
	int32_t nKVHeads = pInput->shape[1];
	int32_t headDim	 = pInput->shape[2];

	lyTensor* pOutput;
	int32_t	  outputShape[] = {seqLen, nKVHeads * nRep, headDim};
	lyCreateTensor(&pOutput, outputShape, 3, NULL, NULL);

	size_t sliceSize = nKVHeads * headDim * sizeof(nv_bfloat16);
	for (int32_t i = 0; i < seqLen; i++)
	{
		for (int32_t j = 0; j < nRep; j++)
		{
			size_t srcOffset = i * sliceSize;
			size_t dstOffset = (i * nKVHeads * nRep + j * nKVHeads) * headDim * sizeof(nv_bfloat16);
			memcpy((uint8_t*)pOutput->data + dstOffset, (uint8_t*)pInput->data + srcOffset, sliceSize);
		}
	}

	*ppOutput = pOutput;
	return true;
}

bool lyUpdateKVCache(lyAttention* pAttention, lyTensor* pK, lyTensor* pV, int32_t startPos)
{
	if (!pAttention || !pK || !pV || startPos < 0)
	{
		return false;
	}

	int32_t seqLen	  = pK->shape[0];
	int32_t maxSeqLen = pAttention->cacheK->shape[0];

	if (startPos + seqLen > maxSeqLen)
	{
		return false;
	}

	int32_t totalElements = seqLen * pAttention->nKVHeads * pAttention->headDim;
	memcpy(pAttention->cacheK->data + startPos * pAttention->nKVHeads * pAttention->headDim, pK->data, totalElements * sizeof(nv_bfloat16));
	memcpy(pAttention->cacheV->data + startPos * pAttention->nKVHeads * pAttention->headDim, pV->data, totalElements * sizeof(nv_bfloat16));

	return true;
}

bool lyAttentionForward(lyTensor** ppOutput, lyAttention* pAttention, lyTensor* pInput, int32_t startPos, lyTensor* pFreqsCis, lyTensor* pMask)
{
	if (!ppOutput || !pAttention || !pInput || !pFreqsCis)
	{
		return false;
	}

	lyTensor *xq, *xk, *xv;
	if (!lyTensorMatMul(&xq, pInput, pAttention->attnWQ))
	{
		return false;
	}

	if (!lyTensorMatMul(&xk, pInput, pAttention->attnWK))
	{
		return false;
	}

	if (!lyTensorMatMul(&xv, pInput, pAttention->attnWV))
	{
		return false;
	}

	int32_t seqLen	 = pInput->shape[0];
	int32_t qShape[] = {seqLen, pAttention->nHeads, pAttention->headDim};
	lyReshapeTensor(xq, qShape, 3);

	int32_t kvShape[] = {seqLen, pAttention->nKVHeads, pAttention->headDim};
	lyReshapeTensor(xk, kvShape, 3);
	lyReshapeTensor(xv, kvShape, 3);

	lyTensor *rotatedQ, *rotatedK;
	if (!lyApplyRotaryEmbedding(&rotatedQ, &rotatedK, xq, xk, pFreqsCis))
	{
		lyDestroyTensor(xq);
		lyDestroyTensor(xk);
		lyDestroyTensor(xv);
		return false;
	}

	lyDestroyTensor(xq);
	lyDestroyTensor(xk);

	if (!lyUpdateKVCache(pAttention, rotatedK, xv, startPos))
	{
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	lyTensor *keys, *values;
	lyTensorSlice(&keys, pAttention->cacheK, 0, startPos + seqLen);
	lyTensorSlice(&values, pAttention->cacheV, 0, startPos + seqLen);

	lyTensor *repeatedKeys, *repeatedValues;
	if (!lyRepeatKV(&repeatedKeys, keys, pAttention->nRep) || !lyRepeatKV(&repeatedValues, values, pAttention->nRep))
	{
		lyDestroyTensor(keys);
		lyDestroyTensor(values);
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	lyDestroyTensor(keys);
	lyDestroyTensor(values);
	lyDestroyTensor(rotatedK);
	lyDestroyTensor(xv);

	lyTensor *transposedQ, *transposedK, *transposedV;
	int32_t	  perm1[] = {1, 0, 2};
	int32_t	  perm2[] = {0, 2, 1};

	if (!lyTensorTranspose(&transposedQ, rotatedQ, perm1) || !lyTensorTranspose(&transposedK, repeatedKeys, perm1) || !lyTensorTranspose(&transposedV, repeatedValues, perm1))
	{
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(repeatedKeys);
		lyDestroyTensor(repeatedValues);
		return false;
	}

	lyDestroyTensor(rotatedQ);
	lyDestroyTensor(repeatedKeys);
	lyDestroyTensor(repeatedValues);

	lyTensor* reTransposedK;
	if (!lyTensorTranspose(&reTransposedK, transposedK, perm2))
	{
		lyDestroyTensor(transposedQ);
		lyDestroyTensor(transposedK);
		lyDestroyTensor(transposedV);
		return false;
	}
	lyDestroyTensor(transposedK);

	lyTensor* scores;
	if (!lyTensorMatMul(&scores, transposedQ, reTransposedK))
	{
		lyDestroyTensor(transposedQ);
		lyDestroyTensor(reTransposedK);
		lyDestroyTensor(transposedV);
		return false;
	}

	float scaleFactor = 1.0f / sqrt(pAttention->headDim);
	if (!lyTensorScaleAndAdd(&scores, scores, pMask, scaleFactor, 1.0f))
	{
		lyDestroyTensor(scores);
		lyDestroyTensor(transposedQ);
		lyDestroyTensor(reTransposedK);
		lyDestroyTensor(transposedV);
		return false;
	}

	lyTensor* attnOut;
	if (!lyTensorMatMul(&attnOut, scores, transposedV))
	{
		lyDestroyTensor(scores);
		lyDestroyTensor(transposedQ);
		lyDestroyTensor(reTransposedK);
		lyDestroyTensor(transposedV);
		return false;
	}

	lyDestroyTensor(scores);
	lyDestroyTensor(transposedQ);
	lyDestroyTensor(reTransposedK);
	lyDestroyTensor(transposedV);

	int32_t flatShape[] = {seqLen, pAttention->nHeads * pAttention->headDim};
	lyReshapeTensor(attnOut, flatShape, 2);
	if (!lyTensorMatMul(ppOutput, attnOut, pAttention->attnWO))
	{
		lyDestroyTensor(attnOut);
		return false;
	}

	lyDestroyTensor(attnOut);
	return true;
}