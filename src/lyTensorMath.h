#pragma once

#include "lyTensor.h"

void lyTensorMatMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
void lyTensorScaleAndAdd(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, float alpha, float beta);
void lyTensorElementwiseMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
void lyTensorMakeTriangularMask(lyTensor* pTensor);
void lyTensorArgmax(lyTensor** ppOutput, const lyTensor* pInput, int32_t dim);
void lyTensorSoftmax(lyTensor** ppOutput, const lyTensor* pInput, int32_t dim);
void lyTensorOuter(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
void lyTensorEmbedding(lyTensor** ppOutput, const lyTensor* pTokens, const lyTensor* pEmbeddings);
void lyTensorTranspose(lyTensor** ppOutput, const lyTensor* pInput, const int32_t* perm);