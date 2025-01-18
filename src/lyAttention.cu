#include "lyAttention.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"

#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

static bool lyUpdateKVCache(lyAttention* pAttention, const lyTensor* pK, const lyTensor* pV, int32_t startPos)
{
	if (!pAttention || !pK || !pV || startPos < 0)
	{
		return false;
	}

	int32_t seqLen = pK->shape[0];

	int32_t destOffset = startPos;
	for (int32_t i = 0; i < seqLen; i++)
	{
		if (destOffset + i >= pAttention->cacheK->shape[0])
		{
			return false;
		}

		size_t		copySize = pAttention->nKVHeads * pAttention->headDim * sizeof(nv_bfloat16);
		cudaError_t error	 = cudaMemcpy(pAttention->cacheK->data + (destOffset + i) * pAttention->nKVHeads * pAttention->headDim, pK->data + i * pAttention->nKVHeads * pAttention->headDim, copySize, cudaMemcpyDeviceToDevice);

		if (error != cudaSuccess)
		{
			return false;
		}

		error = cudaMemcpy(pAttention->cacheV->data + (destOffset + i) * pAttention->nKVHeads * pAttention->headDim, pV->data + i * pAttention->nKVHeads * pAttention->headDim, copySize, cudaMemcpyDeviceToDevice);

		if (error != cudaSuccess)
		{
			return false;
		}
	}

	return true;
}

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
	int32_t dim					= pModel->args.dim;
	int32_t normalHeadsTotalDim = pModel->args.nHeads * pModel->args.headDim;
	int32_t kvHeadsTotalDim		= pModel->args.nKVHeads * pModel->args.headDim;

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wq.weight", layerIndex);
	if (!lyGetModelTensor(&pAttention->attnWQ, pModel, tensorName))
	{
		free(pAttention);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wk.weight", layerIndex);
	if (!lyGetModelTensor(&pAttention->attnWK, pModel, tensorName))
	{
		lyDestroyAttention(pAttention);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wv.weight", layerIndex);
	if (!lyGetModelTensor(&pAttention->attnWV, pModel, tensorName))
	{
		lyDestroyAttention(pAttention);
		return false;
	}

	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wo.weight", layerIndex);
	if (!lyGetModelTensor(&pAttention->attnWO, pModel, tensorName))
	{
		lyDestroyAttention(pAttention);
		return false;
	}

	int32_t cacheShape[] = {pModel->args.maxSequenceLength, pModel->args.nKVHeads, pModel->args.headDim};

	lyTensor* pCacheK;
	if (!lyCreateTensor(&pCacheK))
	{
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lySetTensorShape(pCacheK, cacheShape, 3))
	{
		lyDestroyTensor(pCacheK);
		lyDestroyAttention(pAttention);
		return false;
	}

	lyTensor* pCacheV;
	if (!lyCreateTensor(&pCacheV))
	{
		lyDestroyTensor(pCacheK);
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lySetTensorShape(pCacheV, cacheShape, 3))
	{
		lyDestroyTensor(pCacheV);
		lyDestroyTensor(pCacheK);
		lyDestroyAttention(pAttention);
		return false;
	}

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

bool lyAttentionForward(lyTensor** ppOutput, lyAttention* pAttention, const lyTensor* pInput, int32_t startPos, const lyTensor* pFreqsCis, const lyTensor* pMask)
{
	if (!ppOutput || !pAttention || !pInput || !pFreqsCis)
	{
		return false;
	}

	lyTensor *xq, *xk, *xv;
	if (!lyTensorMatMul(&xq, pInput, pAttention->attnWQ) || !lyTensorMatMul(&xk, pInput, pAttention->attnWK) || !lyTensorMatMul(&xv, pInput, pAttention->attnWV))
	{
		return false;
	}

	int32_t seqLen	 = pInput->shape[0];
	int32_t qShape[] = {seqLen, pAttention->nHeads, pAttention->headDim};
	if (!lyReshapeTensor(xq, qShape, 3))
	{
		lyDestroyTensor(xq);
		lyDestroyTensor(xk);
		lyDestroyTensor(xv);
		return false;
	}

	int32_t kvShape[] = {seqLen, pAttention->nKVHeads, pAttention->headDim};
	if (!lyReshapeTensor(xk, kvShape, 3) || !lyReshapeTensor(xv, kvShape, 3))
	{
		lyDestroyTensor(xq);
		lyDestroyTensor(xk);
		lyDestroyTensor(xv);
		return false;
	}

	lyTensor *rotatedQ, *rotatedK;
	if (!lyApplyRotaryEmbedding(&rotatedQ, &rotatedK, xq, xk, pFreqsCis))
	{
		lyDestroyTensor(xq);
		lyDestroyTensor(xk);
		lyDestroyTensor(xv);
		return false;
	}

	if (!lyUpdateKVCache(pAttention, rotatedK, xv, startPos))
	{
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	lyTensor* scores;
	if (!lyTensorMatMul(&scores, rotatedQ, rotatedK))
	{
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	float scaleFactor = 1.0f / sqrt(pAttention->headDim);
	if (!lyTensorScaleAndAdd(&scores, scores, pMask, scaleFactor))
	{
		lyDestroyTensor(scores);
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	lyTensor* attnOut;
	if (!lyTensorMatMul(&attnOut, scores, xv))
	{
		lyDestroyTensor(scores);
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	if (!lyTensorMatMul(ppOutput, attnOut, pAttention->attnWO))
	{
		lyDestroyTensor(attnOut);
		lyDestroyTensor(scores);
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	lyDestroyTensor(attnOut);
	lyDestroyTensor(scores);
	lyDestroyTensor(rotatedQ);
	lyDestroyTensor(rotatedK);
	lyDestroyTensor(xv);

	return true;
}