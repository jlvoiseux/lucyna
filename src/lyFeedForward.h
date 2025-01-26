#pragma once

#include "lyModel.h"
#include "lyTensor.h"

typedef struct lyFeedForward
{
	int32_t ffnHiddenDim;

	lyTensor* ffnGate;
	lyTensor* ffnDown;
	lyTensor* ffnUp;
} lyFeedForward;

void lyFeedForwardCreate(lyFeedForward** ppFeedForward, const lyModel* pModel, int32_t layerIndex);
void lyFeedForwardDestroy(lyFeedForward* pFeedForward);
void lyFeedForwardForward(lyTensor** ppOutput, const lyFeedForward* pFeedForward, lyTensor* pInput);