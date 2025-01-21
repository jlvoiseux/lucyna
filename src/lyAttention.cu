#include "lyAttention.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"

#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void updateKVCacheKernel(nv_bfloat16* cacheK, nv_bfloat16* cacheV, const nv_bfloat16* k, const nv_bfloat16* v, int32_t startPos, int32_t seqLen, int32_t nKVHeads, int32_t headDim, int32_t maxSeqLen)
{
	int idx			  = blockIdx.x * blockDim.x + threadIdx.x;
	int totalElements = seqLen * nKVHeads * headDim;

	if (idx >= totalElements)
	{
		return;
	}

	int seqIdx	  = idx / (nKVHeads * headDim);
	int remainder = idx % (nKVHeads * headDim);

	int destPos = startPos + seqIdx;
	if (destPos >= maxSeqLen)
	{
		return;
	}

	cacheK[destPos * nKVHeads * headDim + remainder] = k[idx];
	cacheV[destPos * nKVHeads * headDim + remainder] = v[idx];
}

bool lyUpdateKVCache(lyAttention* pAttention, const lyTensor* pK, const lyTensor* pV, int32_t startPos)
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
	int32_t blockSize	  = 256;
	int32_t numBlocks	  = (totalElements + blockSize - 1) / blockSize;

	updateKVCacheKernel<<<numBlocks, blockSize>>>(pAttention->cacheK->data, pAttention->cacheV->data, pK->data, pV->data, startPos, seqLen, pAttention->nKVHeads, pAttention->headDim, maxSeqLen);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		return false;
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

	int32_t perm[] = {1, 0};

	// Get and transpose WQ
	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wq.weight", layerIndex);
	lyTensor* tempWQ;
	if (!lyGetModelTensor(&tempWQ, pModel, tensorName))
	{
		free(pAttention);
		return false;
	}
	if (!lyTensorTranspose(&pAttention->attnWQ, tempWQ, perm))
	{
		free(pAttention);
		return false;
	}

	// Get and transpose WK
	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wk.weight", layerIndex);
	lyTensor* tempWK;
	if (!lyGetModelTensor(&tempWK, pModel, tensorName))
	{
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lyTensorTranspose(&pAttention->attnWK, tempWK, perm))
	{
		lyDestroyAttention(pAttention);
		return false;
	}

	// Get and transpose WV
	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wv.weight", layerIndex);
	lyTensor* tempWV;
	if (!lyGetModelTensor(&tempWV, pModel, tensorName))
	{
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lyTensorTranspose(&pAttention->attnWV, tempWV, perm))
	{
		lyDestroyAttention(pAttention);
		return false;
	}

	// Get and transpose WO
	snprintf(tensorName, sizeof(tensorName), "layers.%d.attention.wo.weight", layerIndex);
	lyTensor* tempWO;
	if (!lyGetModelTensor(&tempWO, pModel, tensorName))
	{
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lyTensorTranspose(&pAttention->attnWO, tempWO, perm))
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

	lyTensor* transposedK;
	int32_t	  permK[] = {1, 2};
	if (!lyTensorTranspose(&transposedK, rotatedK, permK))
	{
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	lyTensor* scores;
	if (!lyTensorMatMul(&scores, rotatedQ, transposedK))
	{
		lyDestroyTensor(transposedK);
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	lyDestroyTensor(transposedK);
	lyDestroyTensor(rotatedQ);
	lyDestroyTensor(rotatedK);

	float scaleFactor = 1.0f / sqrt(pAttention->headDim);
	if (!lyTensorScaleAndAdd(&scores, scores, pMask, scaleFactor))
	{
		lyDestroyTensor(scores);
		lyDestroyTensor(xv);
		return false;
	}

	lyTensor* attnOut;
	if (!lyTensorMatMul(&attnOut, scores, xv))
	{
		lyDestroyTensor(scores);
		lyDestroyTensor(xv);
		return false;
	}

	int32_t flatShape[] = {seqLen, pAttention->nHeads * pAttention->headDim};
	if (!lyReshapeTensor(attnOut, flatShape, 2))
	{
		lyDestroyTensor(attnOut);
		lyDestroyTensor(scores);
		lyDestroyTensor(xv);
		return false;
	}

	if (!lyTensorMatMul(ppOutput, attnOut, pAttention->attnWO))
	{
		lyDestroyTensor(attnOut);
		lyDestroyTensor(scores);
		lyDestroyTensor(xv);
		return false;
	}

	lyDestroyTensor(attnOut);
	lyDestroyTensor(scores);
	lyDestroyTensor(xv);

	return true;
}