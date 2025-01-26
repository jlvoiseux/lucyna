#pragma once

#include "lyTensor.h"

typedef struct lyRMSNorm
{
	float	  epsilon;
	lyTensor* weights;
} lyRMSNorm;

void lyRMSNormCreate(lyRMSNorm** ppNorm, float epsilon, lyTensor* pWeights);
void lyRMSNormDestroy(lyRMSNorm* pNorm);
void lyRMSNormForward(lyTensor** ppOutput, const lyRMSNorm* pNorm, lyTensor* pInput);