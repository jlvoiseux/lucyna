#include "lyAttention.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"

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
		lyDestroyTensor(tempWQ);
		free(pAttention);
		return false;
	}
	lyDestroyTensor(tempWQ);

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
		lyDestroyTensor(tempWK);
		lyDestroyAttention(pAttention);
		return false;
	}
	lyDestroyTensor(tempWK);

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
		lyDestroyTensor(tempWV);
		lyDestroyAttention(pAttention);
		return false;
	}
	lyDestroyTensor(tempWV);

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
		lyDestroyTensor(tempWO);
		lyDestroyAttention(pAttention);
		return false;
	}
	lyDestroyTensor(tempWO);

	int32_t cacheShape[] = {pModel->args.maxSequenceLength, pModel->args.nKVHeads, pModel->args.headDim};

	lyTensor* pCacheK;
	if (!lyCreateTensor(&pCacheK, LY_MEMORY_GPU))
	{
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lySetTensorShape(pCacheK, cacheShape, 3) || !lySetTensorData(pCacheK, NULL, cacheShape[0] * cacheShape[1] * cacheShape[2] * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pCacheK);
		lyDestroyAttention(pAttention);
		return false;
	}

	lyTensor* pCacheV;
	if (!lyCreateTensor(&pCacheV, LY_MEMORY_GPU))
	{
		lyDestroyTensor(pCacheK);
		lyDestroyAttention(pAttention);
		return false;
	}
	if (!lySetTensorShape(pCacheV, cacheShape, 3) || !lySetTensorData(pCacheV, NULL, cacheShape[0] * cacheShape[1] * cacheShape[2] * sizeof(nv_bfloat16)))
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

__global__ void repeatKVKernel(nv_bfloat16* output, const nv_bfloat16* input, int32_t seqLen, int32_t nKVHeads, int32_t nRep, int32_t headDim)
{
	int idx			  = blockIdx.x * blockDim.x + threadIdx.x;
	int totalElements = seqLen * nKVHeads * nRep * headDim;

	if (idx >= totalElements)
	{
		return;
	}

	// Calculate indices
	int headDimIdx = idx % headDim;
	int tmp		   = idx / headDim;
	int repIdx	   = tmp % nRep;
	tmp			   = tmp / nRep;
	int kvHeadIdx  = tmp % nKVHeads;
	int seqIdx	   = tmp / nKVHeads;

	// Source index in input tensor (ignoring rep dimension)
	int srcIdx = seqIdx * (nKVHeads * headDim) + kvHeadIdx * headDim + headDimIdx;

	// Copy value to output
	output[idx] = input[srcIdx];
}

bool lyRepeatKV(lyTensor** ppOutput, const lyTensor* pInput, int32_t nRep)
{
	if (pInput->memoryType == LY_MEMORY_CPU)
	{
		printf("CUDA operations on CPU tensors are not supported");
		return false;
	}

	if (!ppOutput || !pInput || nRep <= 0)
	{
		return false;
	}

	// If no repetition needed, just create a copy
	if (nRep == 1)
	{
		lyTensor* pOutput;
		if (!lyCreateTensor(&pOutput, LY_MEMORY_GPU))
		{
			return false;
		}
		if (!lySetTensorShape(pOutput, pInput->shape, pInput->rank) || !lySetTensorData(pOutput, pInput->data, pInput->dataSize))
		{
			lyDestroyTensor(pOutput);
			return false;
		}
		*ppOutput = pOutput;
		return true;
	}

	// Input shape should be [seqLen, nKVHeads, headDim]
	if (pInput->rank != 3)
	{
		return false;
	}

	int32_t seqLen	 = pInput->shape[0];
	int32_t nKVHeads = pInput->shape[1];
	int32_t headDim	 = pInput->shape[2];

	// Create output tensor with shape [seqLen, nKVHeads * nRep, headDim]
	lyTensor* pOutput;
	if (!lyCreateTensor(&pOutput, LY_MEMORY_GPU))
	{
		return false;
	}

	int32_t outputShape[] = {seqLen, nKVHeads * nRep, headDim};
	if (!lySetTensorShape(pOutput, outputShape, 3) || !lySetTensorData(pOutput, NULL, seqLen * nKVHeads * nRep * headDim * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	// Launch kernel to repeat the heads
	int32_t totalElements = seqLen * nKVHeads * nRep * headDim;
	int32_t blockSize	  = 256;
	int32_t numBlocks	  = (totalElements + blockSize - 1) / blockSize;

	repeatKVKernel<<<numBlocks, blockSize>>>(pOutput->data, pInput->data, seqLen, nKVHeads, nRep, headDim);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pOutput);
		return false;
	}

	*ppOutput = pOutput;
	return true;
}

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
	if (pK->memoryType == LY_MEMORY_CPU || pV->memoryType == LY_MEMORY_CPU)
	{
		printf("CUDA operations on CPU tensors are not supported");
		return false;
	}

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

bool lyAttentionForward(lyTensor** ppOutput, lyAttention* pAttention, const lyTensor* pInput, int32_t startPos, const lyTensor* pFreqsCis, const lyTensor* pMask)
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

	lyDestroyTensor(xq);
	lyDestroyTensor(xk);

	// Update KV cache
	if (!lyUpdateKVCache(pAttention, rotatedK, xv, startPos))
	{
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	// Get cached keys and values
	lyTensor *keys, *values;
	if (!lyTensorSlice(&keys, pAttention->cacheK, 0, startPos + seqLen) || !lyTensorSlice(&values, pAttention->cacheV, 0, startPos + seqLen))
	{
		lyDestroyTensor(rotatedQ);
		lyDestroyTensor(rotatedK);
		lyDestroyTensor(xv);
		return false;
	}

	// Repeat KV heads if needed
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

	// Transpose for matrix multiplication
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

	// Calculate attention scores
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

	// Compute attention output
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

	// Reshape output
	int32_t flatShape[] = {seqLen, pAttention->nHeads * pAttention->headDim};
	if (!lyReshapeTensor(attnOut, flatShape, 2))
	{
		lyDestroyTensor(attnOut);
		return false;
	}

	// Final linear transformation
	if (!lyTensorMatMul(ppOutput, attnOut, pAttention->attnWO))
	{
		lyDestroyTensor(attnOut);
		return false;
	}

	lyDestroyTensor(attnOut);
	return true;
}