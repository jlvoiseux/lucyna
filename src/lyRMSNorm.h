#pragma once

#include "lyTensor.h"

#include <lyOpenCL.h>

typedef struct lyRMSNorm
{
	float			 epsilon;
	lyTensor*		 weights;
	lyOpenCLContext* openCLContext;
} lyRMSNorm;

void lyRMSNormCreate(lyRMSNorm** ppNorm, float epsilon, lyTensor* pWeights, lyOpenCLContext* pContext);
void lyRMSNormDestroy(lyRMSNorm* pNorm);
void lyRMSNormForward(lyTensor** ppOutput, const lyRMSNorm* pNorm, lyTensor* pInput);