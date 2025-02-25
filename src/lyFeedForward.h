#pragma once

#include "lyModel.h"
#include "lyOpenCL.h"
#include "lyTensor.h"

typedef struct lyFeedForward
{
	int32_t ffnHiddenDim;

	lyTensor* ffnGate;
	lyTensor* ffnDown;
	lyTensor* ffnUp;

	lyOpenCLContext* openCLContext;
} lyFeedForward;

void lyFeedForwardCreate(lyFeedForward** ppFeedForward, const lyModel* pModel, int32_t layerIndex, lyOpenCLContext* pContext);

void lyFeedForwardDestroy(lyFeedForward* pFeedForward);
void lyFeedForwardForward(lyTensor** ppOutput, const lyFeedForward* pFeedForward, lyTensor* pInput);