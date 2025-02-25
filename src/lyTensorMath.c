#include "lyTensorMath.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lyTensorMatMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, lyOpenCLContext* pContext)
{
	if (!pContext || !pContext->initialized || !pContext->kernels.initialized)
	{
		fprintf(stderr, "OpenCL context or kernels not initialized\n");
		return;
	}

	int32_t m = pA->shape[pA->rank - 2];
	int32_t k = pA->shape[pA->rank - 1];
	int32_t n = pB->shape[pB->rank - 1];

	int32_t batchSize = 1;
	for (int32_t i = 0; i < pA->rank - 2; i++)
	{
		batchSize *= pA->shape[i];
	}

	int32_t* outShape = (int32_t*)malloc(sizeof(int32_t) * pA->rank);
	for (int32_t i = 0; i < pA->rank - 2; i++)
	{
		outShape[i] = pA->shape[i];
	}
	outShape[pA->rank - 2] = m;
	outShape[pA->rank - 1] = n;

	lyTensor* pOutput;
	lyTensorCreate(&pOutput, outShape, pA->rank, NULL, NULL);
	free(outShape);

	size_t matrixSizeA = m * k * sizeof(lyBfloat16);
	size_t matrixSizeB = k * n * sizeof(lyBfloat16);
	size_t matrixSizeC = m * n * sizeof(float);	 // Float intermediate storage

	float* tempOutput = (float*)malloc(matrixSizeC);

	for (int32_t batch = 0; batch < batchSize; batch++)
	{
		lyBfloat16* pBatchA	  = pA->data + batch * m * k;
		lyBfloat16* pBatchB	  = pB->data + batch * k * n;
		lyBfloat16* pBatchOut = pOutput->data + batch * m * n;

		cl_int status;
		cl_mem bufferA = clCreateBuffer(pContext->context, CL_MEM_READ_ONLY, matrixSizeA, NULL, &status);
		cl_mem bufferB = clCreateBuffer(pContext->context, CL_MEM_READ_ONLY, matrixSizeB, NULL, &status);
		cl_mem bufferC = clCreateBuffer(pContext->context, CL_MEM_WRITE_ONLY, matrixSizeC, NULL, &status);

		status = clEnqueueWriteBuffer(pContext->queue, bufferA, CL_TRUE, 0, matrixSizeA, pBatchA, 0, NULL, NULL);
		status |= clEnqueueWriteBuffer(pContext->queue, bufferB, CL_TRUE, 0, matrixSizeB, pBatchB, 0, NULL, NULL);

		if (status != CL_SUCCESS)
		{
			fprintf(stderr, "Failed to write to OpenCL buffers: %d\n", status);
			clReleaseMemObject(bufferA);
			clReleaseMemObject(bufferB);
			clReleaseMemObject(bufferC);
			free(tempOutput);
			lyTensorDestroy(pOutput);
			return;
		}

		cl_kernel kernel = pContext->kernels.matMulKernel;

		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
		status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
		status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
		status |= clSetKernelArg(kernel, 3, sizeof(int), &m);
		status |= clSetKernelArg(kernel, 4, sizeof(int), &n);
		status |= clSetKernelArg(kernel, 5, sizeof(int), &k);

		if (status != CL_SUCCESS)
		{
			fprintf(stderr, "Failed to set kernel arguments: %d\n", status);
			clReleaseMemObject(bufferA);
			clReleaseMemObject(bufferB);
			clReleaseMemObject(bufferC);
			free(tempOutput);
			lyTensorDestroy(pOutput);
			return;
		}

		size_t globalWorkSize[2] = {n, m};

		status = clEnqueueNDRangeKernel(pContext->queue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);

		if (status != CL_SUCCESS)
		{
			fprintf(stderr, "Failed to execute kernel: %d\n", status);
			clReleaseMemObject(bufferA);
			clReleaseMemObject(bufferB);
			clReleaseMemObject(bufferC);
			free(tempOutput);
			lyTensorDestroy(pOutput);
			return;
		}

		status = clEnqueueReadBuffer(pContext->queue, bufferC, CL_TRUE, 0, matrixSizeC, tempOutput, 0, NULL, NULL);

		if (status != CL_SUCCESS)
		{
			fprintf(stderr, "Failed to read results: %d\n", status);
			clReleaseMemObject(bufferA);
			clReleaseMemObject(bufferB);
			clReleaseMemObject(bufferC);
			free(tempOutput);
			lyTensorDestroy(pOutput);
			return;
		}

		for (int i = 0; i < m * n; i++)
		{
			pBatchOut[i] = lyFloat32ToBfloat16(tempOutput[i]);
		}

		clReleaseMemObject(bufferA);
		clReleaseMemObject(bufferB);
		clReleaseMemObject(bufferC);
	}

	free(tempOutput);
	*ppOutput = pOutput;
}

void lyTensorScaleAndAdd(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, lyBfloat16 alpha, lyBfloat16 beta, lyOpenCLContext* pContext)
{
	if (!pContext || !pContext->initialized || !pContext->kernels.initialized)
	{
		fprintf(stderr, "OpenCL context or kernels not initialized\n");
		return;
	}

	lyTensor* pOutput;
	lyTensorCreate(&pOutput, pA->shape, pA->rank, NULL, NULL);

	int32_t totalElements = 1;
	for (int32_t i = 0; i < pA->rank; i++)
	{
		totalElements *= pA->shape[i];
	}

	int32_t* aStrides = (int32_t*)malloc(pA->rank * sizeof(int32_t));
	int32_t* bStrides = NULL;

	aStrides[pA->rank - 1] = 1;
	for (int32_t i = pA->rank - 2; i >= 0; i--)
	{
		aStrides[i] = aStrides[i + 1] * pA->shape[i + 1];
	}

	int32_t bRank = 0;
	if (pB)
	{
		bRank				   = pB->rank;
		bStrides			   = (int32_t*)malloc(pB->rank * sizeof(int32_t));
		bStrides[pB->rank - 1] = 1;
		for (int32_t i = pB->rank - 2; i >= 0; i--)
		{
			bStrides[i] = bStrides[i + 1] * pB->shape[i + 1];
		}
	}

	size_t dataSizeA = totalElements * sizeof(lyBfloat16);

	cl_int status;
	cl_mem bufferA		  = clCreateBuffer(pContext->context, CL_MEM_READ_ONLY, dataSizeA, NULL, &status);
	cl_mem bufferOutput	  = clCreateBuffer(pContext->context, CL_MEM_WRITE_ONLY, dataSizeA, NULL, &status);
	cl_mem bufferAStrides = clCreateBuffer(pContext->context, CL_MEM_READ_ONLY, pA->rank * sizeof(int32_t), NULL, &status);
	cl_mem bufferAShape	  = clCreateBuffer(pContext->context, CL_MEM_READ_ONLY, pA->rank * sizeof(int32_t), NULL, &status);

	status = clEnqueueWriteBuffer(pContext->queue, bufferA, CL_TRUE, 0, dataSizeA, pA->data, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(pContext->queue, bufferAStrides, CL_TRUE, 0, pA->rank * sizeof(int32_t), aStrides, 0, NULL, NULL);
	status |= clEnqueueWriteBuffer(pContext->queue, bufferAShape, CL_TRUE, 0, pA->rank * sizeof(int32_t), pA->shape, 0, NULL, NULL);

	cl_mem bufferB		  = NULL;
	cl_mem bufferBStrides = NULL;

	if (pB)
	{
		size_t dataSizeB = 1;
		for (int32_t i = 0; i < pB->rank; i++)
		{
			dataSizeB *= pB->shape[i];
		}
		dataSizeB *= sizeof(lyBfloat16);

		bufferB		   = clCreateBuffer(pContext->context, CL_MEM_READ_ONLY, dataSizeB, NULL, &status);
		bufferBStrides = clCreateBuffer(pContext->context, CL_MEM_READ_ONLY, pB->rank * sizeof(int32_t), NULL, &status);

		status |= clEnqueueWriteBuffer(pContext->queue, bufferB, CL_TRUE, 0, dataSizeB, pB->data, 0, NULL, NULL);
		status |= clEnqueueWriteBuffer(pContext->queue, bufferBStrides, CL_TRUE, 0, pB->rank * sizeof(int32_t), bStrides, 0, NULL, NULL);
	}

	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to write to OpenCL buffers: %d\n", status);
		clReleaseMemObject(bufferA);
		clReleaseMemObject(bufferOutput);
		clReleaseMemObject(bufferAStrides);
		clReleaseMemObject(bufferAShape);
		if (bufferB)
			clReleaseMemObject(bufferB);
		if (bufferBStrides)
			clReleaseMemObject(bufferBStrides);
		free(aStrides);
		if (bStrides)
			free(bStrides);
		lyTensorDestroy(pOutput);
		return;
	}

	cl_kernel kernel = pContext->kernels.scaleAndAddKernel;

	float alphaF = lyBfloat16ToFloat32(alpha);
	float betaF	 = lyBfloat16ToFloat32(beta);

	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), pB ? &bufferB : NULL);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferOutput);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferAStrides);
	status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), pB ? &bufferBStrides : NULL);
	status |= clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufferAShape);
	status |= clSetKernelArg(kernel, 6, sizeof(int), &pA->rank);
	status |= clSetKernelArg(kernel, 7, sizeof(int), &bRank);
	status |= clSetKernelArg(kernel, 8, sizeof(float), &alphaF);
	status |= clSetKernelArg(kernel, 9, sizeof(float), &betaF);
	status |= clSetKernelArg(kernel, 10, sizeof(int), &totalElements);

	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to set kernel arguments: %d\n", status);
		clReleaseMemObject(bufferA);
		clReleaseMemObject(bufferOutput);
		clReleaseMemObject(bufferAStrides);
		clReleaseMemObject(bufferAShape);
		if (bufferB)
			clReleaseMemObject(bufferB);
		if (bufferBStrides)
			clReleaseMemObject(bufferBStrides);
		free(aStrides);
		if (bStrides)
			free(bStrides);
		lyTensorDestroy(pOutput);
		return;
	}

	size_t globalWorkSize = totalElements;
	status				  = clEnqueueNDRangeKernel(pContext->queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);

	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to execute kernel: %d\n", status);
		clReleaseMemObject(bufferA);
		clReleaseMemObject(bufferOutput);
		clReleaseMemObject(bufferAStrides);
		clReleaseMemObject(bufferAShape);
		if (bufferB)
			clReleaseMemObject(bufferB);
		if (bufferBStrides)
			clReleaseMemObject(bufferBStrides);
		free(aStrides);
		if (bStrides)
			free(bStrides);
		lyTensorDestroy(pOutput);
		return;
	}

	status = clEnqueueReadBuffer(pContext->queue, bufferOutput, CL_TRUE, 0, dataSizeA, pOutput->data, 0, NULL, NULL);

	if (status != CL_SUCCESS)
	{
		fprintf(stderr, "Failed to read results: %d\n", status);
		clReleaseMemObject(bufferA);
		clReleaseMemObject(bufferOutput);
		clReleaseMemObject(bufferAStrides);
		clReleaseMemObject(bufferAShape);
		if (bufferB)
			clReleaseMemObject(bufferB);
		if (bufferBStrides)
			clReleaseMemObject(bufferBStrides);
		free(aStrides);
		if (bStrides)
			free(bStrides);
		lyTensorDestroy(pOutput);
		return;
	}

	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferOutput);
	clReleaseMemObject(bufferAStrides);
	clReleaseMemObject(bufferAShape);
	if (bufferB)
		clReleaseMemObject(bufferB);
	if (bufferBStrides)
		clReleaseMemObject(bufferBStrides);

	free(aStrides);
	if (bStrides)
		free(bStrides);

	*ppOutput = pOutput;
}

void lyTensorElementwiseMul(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, lyOpenCLContext* pOpenCLContext)
{
	lyTensor* pOutput;
	lyTensorCreate(&pOutput, pA->shape, pA->rank, NULL, NULL);

	int32_t* aStrides = (int32_t*)malloc(pA->rank * sizeof(int32_t));
	int32_t* bStrides = (int32_t*)malloc(pB->rank * sizeof(int32_t));

	aStrides[pA->rank - 1] = 1;
	for (int32_t i = pA->rank - 2; i >= 0; i--)
	{
		aStrides[i] = aStrides[i + 1] * pA->shape[i + 1];
	}

	bStrides[pB->rank - 1] = 1;
	for (int32_t i = pB->rank - 2; i >= 0; i--)
	{
		bStrides[i] = bStrides[i + 1] * pB->shape[i + 1];
	}

	int32_t totalElements = 1;
	for (int32_t i = 0; i < pA->rank; i++)
	{
		totalElements *= pA->shape[i];
	}

	for (int32_t i = 0; i < totalElements; i++)
	{
		int32_t bIdx = 0;
		int32_t temp = i;

		for (int32_t j = 0; j < pB->rank; j++)
		{
			int32_t aRankOffset = pA->rank - pB->rank + j;
			int32_t dimIdx		= (temp / aStrides[aRankOffset]) % pA->shape[aRankOffset];
			bIdx += dimIdx * bStrides[j];
			temp %= aStrides[aRankOffset];
		}

		pOutput->data[i] = lyFloat32ToBfloat16(lyBfloat16ToFloat32(pA->data[i]) * lyBfloat16ToFloat32(pB->data[bIdx]));
	}

	free(aStrides);
	free(bStrides);

	*ppOutput = pOutput;
}

void lyTensorMakeTriangularMask(lyTensor* pTensor, lyOpenCLContext* pOpenCLContext)
{
	int32_t rows = pTensor->shape[0];
	int32_t cols = pTensor->shape[1];

	for (int32_t row = 0; row < rows; row++)
	{
		for (int32_t col = 0; col < cols; col++)
		{
			float val						= col <= row ? 0.0f : -HUGE_VALF;
			pTensor->data[row * cols + col] = lyFloat32ToBfloat16(val);
		}
	}
}

void lyTensorSoftmax(lyTensor** ppOutput, const lyTensor* pInput, lyOpenCLContext* pOpenCLContext)
{
	int32_t	  dim = pInput->rank - 1;
	lyTensor* pOutput;
	lyTensorCreate(&pOutput, pInput->shape, pInput->rank, NULL, NULL);

	int32_t dimSize	  = pInput->shape[dim];
	int32_t outerSize = 1;
	for (int32_t i = 0; i < dim; i++)
	{
		outerSize *= pInput->shape[i];
	}

	for (int32_t i = 0; i < outerSize; i++)
	{
		int32_t rowOffset = i * dimSize;

		float rowMax = -FLT_MAX;
		for (int32_t j = 0; j < dimSize; j++)
		{
			float val = lyBfloat16ToFloat32(pInput->data[rowOffset + j]);
			if (val > rowMax)
				rowMax = val;
		}

		float rowExpSum = 0.0f;
		for (int32_t j = 0; j < dimSize; j++)
		{
			float val = lyBfloat16ToFloat32(pInput->data[rowOffset + j]);
			rowExpSum += expf(val - rowMax);
		}

		for (int32_t j = 0; j < dimSize; j++)
		{
			float val					 = lyBfloat16ToFloat32(pInput->data[rowOffset + j]);
			float outVal				 = expf(val - rowMax) / rowExpSum;
			pOutput->data[rowOffset + j] = lyFloat32ToBfloat16(outVal);
		}
	}

	*ppOutput = pOutput;
}

void lyTensorArgmax(int32_t* pOutput, const lyTensor* pInput, lyOpenCLContext* pOpenCLContext)
{
	if (pInput->rank > 2 || (pInput->rank == 2 && pInput->shape[0] != 1 && pInput->shape[1] != 1))
	{
		*pOutput = -1;
		return;
	}

	int32_t vectorLength = pInput->rank == 1 ? pInput->shape[0] : (pInput->shape[0] == 1 ? pInput->shape[1] : pInput->shape[0]);

	float	maxVal = -HUGE_VALF;
	int32_t maxIdx = 0;

	for (int32_t i = 0; i < vectorLength; i++)
	{
		float val = lyBfloat16ToFloat32(pInput->data[i]);
		if (val > maxVal)
		{
			maxVal = val;
			maxIdx = i;
		}
	}

	*pOutput = maxIdx;
}

void lyTensorOuter(lyTensor** ppOutput, const lyTensor* pA, const lyTensor* pB, lyOpenCLContext* pOpenCLContext)
{
	lyTensor* pOutput;
	int32_t	  outputShape[] = {pA->shape[0], pB->shape[0]};
	lyTensorCreate(&pOutput, outputShape, 2, NULL, NULL);

	for (int32_t i = 0; i < pA->shape[0]; i++)
	{
		for (int32_t j = 0; j < pB->shape[0]; j++)
		{
			float aVal							= lyBfloat16ToFloat32(pA->data[i]);
			float bVal							= lyBfloat16ToFloat32(pB->data[j]);
			pOutput->data[i * pB->shape[0] + j] = lyFloat32ToBfloat16(aVal * bVal);
		}
	}

	*ppOutput = pOutput;
}

void lyTensorEmbedding(lyTensor** ppOutput, const int32_t* pInputTokens, int32_t seqLen, const lyTensor* pEmbeddings, lyOpenCLContext* pOpenCLContext)
{
	if (pEmbeddings->rank != 2)
	{
		return;
	}

	int32_t	  embeddingDim = pEmbeddings->shape[1];
	lyTensor* pOutput;
	int32_t	  outputShape[] = {seqLen, embeddingDim};
	lyTensorCreate(&pOutput, outputShape, 2, NULL, NULL);
	size_t rowSizeBytes = embeddingDim * sizeof(lyBfloat16);

	for (int32_t rowIdx = 0; rowIdx < seqLen; rowIdx++)
	{
		int32_t tokenId	  = pInputTokens[rowIdx];
		size_t	srcOffset = tokenId * embeddingDim;
		size_t	dstOffset = rowIdx * embeddingDim;
		memcpy(pOutput->data + dstOffset, pEmbeddings->data + srcOffset, rowSizeBytes);
	}

	*ppOutput = pOutput;
}

void lyTensorTranspose(lyTensor** ppOutput, const lyTensor* pInput, const int32_t* pPerm, lyOpenCLContext* pOpenCLContext)
{
	int32_t* newShape = (int32_t*)malloc(sizeof(int32_t) * pInput->rank);
	for (int32_t i = 0; i < pInput->rank; i++)
	{
		newShape[i] = pInput->shape[pPerm[i]];
	}

	lyTensor* pOutput;
	lyTensorCreate(&pOutput, newShape, pInput->rank, NULL, NULL);
	free(newShape);

	if (pInput->rank == 2)
	{
		int32_t rows = pInput->shape[0];
		int32_t cols = pInput->shape[1];

		for (int32_t i = 0; i < rows; i++)
		{
			for (int32_t j = 0; j < cols; j++)
			{
				int32_t inIdx = i * cols + j;
				int32_t outIdx;

				if (pPerm[0] == 0 && pPerm[1] == 1)
					outIdx = inIdx;
				else if (pPerm[0] == 1 && pPerm[1] == 0)
					outIdx = j * rows + i;
				else
					return;

				pOutput->data[outIdx] = pInput->data[inIdx];
			}
		}
	}
	else if (pInput->rank == 3)
	{
		int32_t inputStrides[3]	 = {0};
		int32_t outputStrides[3] = {0};

		inputStrides[pInput->rank - 1] = 1;
		for (int32_t i = pInput->rank - 2; i >= 0; i--)
		{
			inputStrides[i] = inputStrides[i + 1] * pInput->shape[i + 1];
		}

		outputStrides[pInput->rank - 1] = 1;
		for (int32_t i = pInput->rank - 2; i >= 0; i--)
		{
			outputStrides[i] = outputStrides[i + 1] * pOutput->shape[i + 1];
		}

		int32_t indices[3] = {0};
		int32_t inputIdx = 0, outputIdx = 0;

		for (int32_t i = 0; i < pInput->shape[0]; i++)
		{
			for (int32_t j = 0; j < pInput->shape[1]; j++)
			{
				for (int32_t k = 0; k < pInput->shape[2]; k++)
				{
					indices[0] = i;
					indices[1] = j;
					indices[2] = k;

					inputIdx  = indices[0] * inputStrides[0] + indices[1] * inputStrides[1] + indices[2] * inputStrides[2];
					outputIdx = indices[pPerm[0]] * outputStrides[0] + indices[pPerm[1]] * outputStrides[1] + indices[pPerm[2]] * outputStrides[2];

					pOutput->data[outputIdx] = pInput->data[inputIdx];
				}
			}
		}
	}

	*ppOutput = pOutput;
}