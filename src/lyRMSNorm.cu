#include "lyRMSNorm.h"
#include "lyTensorMath.h"

#include <cuda_bf16.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

void lyRMSNormCreate(lyRMSNorm** ppNorm, float epsilon, lyTensor* pWeights)
{
	lyRMSNorm* pNorm = (lyRMSNorm*)malloc(sizeof(lyRMSNorm));

	pNorm->epsilon = epsilon;
	pNorm->weights = pWeights;
	*ppNorm		   = pNorm;
}

void lyRMSNormDestroy(lyRMSNorm* pNorm)
{
	if (!pNorm)
		return;

	free(pNorm);
}

static bool doNormalization(lyTensor** ppOutput, const lyRMSNorm* pNorm, lyTensor* pInput)
{
	int seqLen = pInput->shape[0];
	int dim	   = pInput->shape[1];

	lyTensorFloat* squared;
	lyTensorFloatCreate(&squared, pInput->shape, pInput->rank, NULL, NULL);
	for (int seqIdx = 0; seqIdx < seqLen; seqIdx++)
	{
		for (int dimIdx = 0; dimIdx < dim; dimIdx++)
		{
			float val							 = __bfloat162float(pInput->data[seqIdx * dim + dimIdx]);
			squared->data[seqIdx * dim + dimIdx] = float((double)val * (double)val);
		}
	}

	int32_t		   meanShape[] = {seqLen, 1};  // keepdim=true
	lyTensorFloat* means;
	lyTensorFloatCreate(&means, meanShape, 2, NULL, NULL);
	for (int seqIdx = 0; seqIdx < seqLen; seqIdx++)
	{
		float sum = 0.0f;
		for (int dimIdx = 0; dimIdx < dim; dimIdx++)
		{
			sum += squared->data[seqIdx * dim + dimIdx];
		}
		means->data[seqIdx] = sum / (float)dim;
	}
	lyTensorFloatDestroy(squared);

	lyTensorFloat* varianceEps;
	lyTensorFloatCreate(&varianceEps, meanShape, 2, NULL, NULL);
	for (int seqIdx = 0; seqIdx < seqLen; seqIdx++)
	{
		float val				  = means->data[seqIdx];
		varianceEps->data[seqIdx] = val + pNorm->epsilon;
	}
	lyTensorFloatDestroy(means);

	lyTensorFloat* invStd;
	lyTensorFloatCreate(&invStd, meanShape, 2, NULL, NULL);
	for (int seqIdx = 0; seqIdx < seqLen; seqIdx++)
	{
		float val			 = varianceEps->data[seqIdx];
		invStd->data[seqIdx] = (float)(1.0 / sqrt((double)val));
	}
	lyTensorFloatDestroy(varianceEps);

	lyTensorCreate(ppOutput, pInput->shape, pInput->rank, NULL, NULL);
	for (int seqIdx = 0; seqIdx < seqLen; seqIdx++)
	{
		float scale = invStd->data[seqIdx];
		for (int dimIdx = 0; dimIdx < dim; dimIdx++)
		{
			float val								 = __bfloat162float(pInput->data[seqIdx * dim + dimIdx]);
			(*ppOutput)->data[seqIdx * dim + dimIdx] = __float2bfloat16_rz(val * scale);
		}
	}
	lyTensorFloatDestroy(invStd);

	return true;
}

void lyRMSNormForward(lyTensor** ppOutput, const lyRMSNorm* pNorm, lyTensor* pInput)
{
	lyTensor* h = NULL;
	doNormalization(&h, pNorm, pInput);
	lyTensorPrint(h);

	lyTensorElementwiseMul(ppOutput, h, pNorm->weights);
	lyTensorPrint(*ppOutput);

	lyTensorDestroy(h);
}
