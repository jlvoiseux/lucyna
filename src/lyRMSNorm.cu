#include "lyRMSNorm.h"

#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

bool lyCreateRMSNorm(lyRMSNorm** ppNorm, float epsilon, lyTensor* pWeights)
{
	if (!ppNorm || !pWeights)
	{
		return false;
	}

	lyRMSNorm* pNorm = (lyRMSNorm*)malloc(sizeof(lyRMSNorm));
	if (!pNorm)
	{
		return false;
	}

	pNorm->epsilon = epsilon;
	pNorm->weights = pWeights;
	*ppNorm		   = pNorm;

	return true;
}

void lyDestroyRMSNorm(lyRMSNorm* pNorm)
{
	if (!pNorm)
	{
		return;
	}

	free(pNorm);
}

bool lyRMSNormForward(lyTensor** ppOutput, const lyRMSNorm* pNorm, lyTensor* pInput)
{
	if (!ppOutput || !pNorm || !pInput || !pInput->data || !pNorm->weights || !pInput->rank)
	{
		return false;
	}

	lyTensor* pOutput;
	lyCreateTensor(&pOutput, pInput->shape, pInput->rank, NULL, NULL);

	int seqLen = pInput->shape[0];
	int dim	   = pInput->shape[1];

	for (int seqIdx = 0; seqIdx < seqLen; seqIdx++)
	{
		float sumSquare = 0.0f;
		for (int dimIdx = 0; dimIdx < dim; dimIdx++)
		{
			float val = __bfloat162float(pInput->data[seqIdx * dim + dimIdx]);
			sumSquare += val * val;
		}

		float meanSquare = sumSquare / (float)dim;
		float scale		 = 1.0f / sqrtf(meanSquare + pNorm->epsilon);

		for (int dimIdx = 0; dimIdx < dim; dimIdx++)
		{
			float val							 = __bfloat162float(pInput->data[seqIdx * dim + dimIdx]);
			float weight						 = __bfloat162float(pNorm->weights->data[dimIdx]);
			pOutput->data[seqIdx * dim + dimIdx] = __float2bfloat16(val * scale * weight);
		}
	}

	*ppOutput = pOutput;
	return true;
}