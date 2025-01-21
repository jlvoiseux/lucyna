#pragma once

#include "lyTensor.h"

bool lyTensorAdd(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
bool lyTensorMatMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
bool lyTensorScaleAndAdd(lyTensor** ppOutput, lyTensor* pInput, const lyTensor* pMask, float scale);
bool lyTensorElementwiseMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
bool lyTensorMakeTriangularMask(lyTensor* pTensor);
bool lyTensorArgmax(lyTensor** ppOutput, const lyTensor* pInput, int32_t dim);
bool lyTensorOuter(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
bool lyTensorEmbedding(lyTensor** ppOutput, const lyTensor* pTokens, const lyTensor* pEmbeddings);
bool lyTensorTranspose(lyTensor** ppOutput, const lyTensor* pInput, const int32_t* perm);