#pragma once

#include "lyTensor.h"

void lyTensorMatMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
void lyTensorScaleAndAdd(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, nv_bfloat16 alpha, nv_bfloat16 beta);
void lyTensorElementwiseMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
void lyTensorMakeTriangularMask(lyTensor* pTensor);
void lyTensorArgmax(int32_t* pOutput, const lyTensor* pInput);
void lyTensorSoftmax(lyTensor** ppOutput, const lyTensor* pInput);
void lyTensorOuter(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB);
void lyTensorEmbedding(lyTensor** ppOutput, const int32_t* pInputTokens, int32_t seqLen, const lyTensor* pEmbeddings);
void lyTensorTranspose(lyTensor** ppOutput, const lyTensor* pInput, const int32_t* perm);

void lyTensorFloatSoftmax(lyTensorFloat** ppOutput, const lyTensorFloat* pInput);