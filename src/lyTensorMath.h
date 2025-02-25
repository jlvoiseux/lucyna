#pragma once

#include "lyTensor.h"

#include <lyOpenCL.h>

void lyTensorMatMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, lyOpenCLContext* pContext);
void lyTensorScaleAndAdd(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, lyBfloat16 alpha, lyBfloat16 beta, lyOpenCLContext* pContext);
void lyTensorElementwiseMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, lyOpenCLContext* pContext);
void lyTensorMakeTriangularMask(lyTensor* pTensor, lyOpenCLContext* pContext);
void lyTensorArgmax(int32_t* pOutput, const lyTensor* pInput, lyOpenCLContext* pContext);
void lyTensorSoftmax(lyTensor** ppOutput, const lyTensor* pInput, lyOpenCLContext* pContext);
void lyTensorOuter(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, lyOpenCLContext* pContext);
void lyTensorEmbedding(lyTensor** ppOutput, const int32_t* pInputTokens, int32_t seqLen, const lyTensor* pEmbeddings, lyOpenCLContext* pContext);
void lyTensorTranspose(lyTensor** ppOutput, const lyTensor* pInput, const int32_t* perm, lyOpenCLContext* pContext);

void lyTensorFloatSoftmax(lyTensorFloat** ppOutput, const lyTensorFloat* pInput, lyOpenCLContext* pContext);