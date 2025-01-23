#pragma once

#include "lyTensor.h"

bool lyTensorMatMul(lyTensor** ppOutput, lyTensor* pA, lyTensor* pB);
bool lyTensorScaleAndAdd(lyTensor** ppOutput, lyTensor* pA, lyTensor* pB, float alpha, float beta);
bool lyTensorElementwiseMul(lyTensor** ppOutput, lyTensor* pA, lyTensor* pB);
bool lyTensorMakeTriangularMask(lyTensor* pTensor);
bool lyTensorArgmax(lyTensor** ppOutput, lyTensor* pInput, int32_t dim);
bool lyTensorOuter(lyTensor** ppOutput, lyTensor* pA, lyTensor* pB);
bool lyTensorEmbedding(lyTensor** ppOutput, lyTensor* pTokens, lyTensor* pEmbeddings);
bool lyTensorTranspose(lyTensor** ppOutput, lyTensor* pInput, int32_t* perm);