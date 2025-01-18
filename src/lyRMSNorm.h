#pragma once

#include "lyTensor.h"

typedef struct lyRMSNorm
{
	float	  epsilon;
	lyTensor* weights;
} lyRMSNorm;

bool lyCreateRMSNorm(lyRMSNorm** ppNorm, float epsilon, lyTensor* pWeights);
void lyDestroyRMSNorm(lyRMSNorm* pNorm);
bool lyRMSNormForward(lyTensor** ppOutput, const lyRMSNorm* pNorm, const lyTensor* pInput);