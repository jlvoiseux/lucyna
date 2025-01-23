#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"

#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

__global__ void computeFrequenciesKernel(nv_bfloat16* output, int32_t dim, int32_t end, float theta)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= dim / 2)
		return;

	float dimFloat = (float)dim;
	float val	   = 1.0f / powf(theta, (2.0f * idx) / dimFloat);

	// Apply scaling inline
	float wavelen = 2.0f * M_PI / val;
	float newVal;

	const float scaleFactor		= 8.0f;
	const float lowFreqFactor	= 1.0f;
	const float highFreqFactor	= 4.0f;
	const float oldContextLen	= 8192.0f;
	const float lowFreqWavelen	= oldContextLen / lowFreqFactor;
	const float highFreqWavelen = oldContextLen / highFreqFactor;

	if (wavelen < highFreqWavelen)
	{
		newVal = val;
	}
	else if (wavelen > lowFreqWavelen)
	{
		newVal = val / scaleFactor;
	}
	else
	{
		float smooth = (oldContextLen / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
		newVal		 = (1.0f - smooth) * val / scaleFactor + smooth * val;
	}

	output[idx] = __float2bfloat16(newVal);
}

__global__ void computeRotaryKernel(nv_bfloat16* output, const nv_bfloat16* freqs, int32_t dim, int32_t end)
{
	int row = blockIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row >= end || col >= dim / 2)
		return;

	float freq	= __bfloat162float(freqs[col]);
	float angle = row * freq;

	// Each complex number takes 2 slots - real part at even indices, imaginary at odd indices
	int baseIdx			= row * dim + 2 * col;	// Note: dim not dim/2 here because we need 2 slots per complex number
	output[baseIdx]		= __float2bfloat16(cosf(angle));
	output[baseIdx + 1] = __float2bfloat16(sinf(angle));
}

__global__ void applyRotaryEmbeddingKernel(nv_bfloat16* xqOut, nv_bfloat16* xkOut, const nv_bfloat16* xq, const nv_bfloat16* xk, const nv_bfloat16* freqsCosReal, const nv_bfloat16* freqsSinReal, int batchSize, int headDim)
{
	int idx		  = blockIdx.x * blockDim.x + threadIdx.x;
	int totalSize = batchSize * headDim / 2;

	if (idx >= totalSize)
	{
		return;
	}

	int row = idx / (headDim / 2);
	int col = (idx % (headDim / 2)) * 2;

	nv_bfloat16 xq_real = xq[row * headDim + col];
	nv_bfloat16 xq_imag = xq[row * headDim + col + 1];
	nv_bfloat16 xk_real = xk[row * headDim + col];
	nv_bfloat16 xk_imag = xk[row * headDim + col + 1];

	nv_bfloat16 cos_real = freqsCosReal[col / 2];
	nv_bfloat16 sin_real = freqsSinReal[col / 2];

	nv_bfloat16 xq_out_real = __hsub(__hmul(xq_real, cos_real), __hmul(xq_imag, sin_real));
	nv_bfloat16 xq_out_imag = __hadd(__hmul(xq_real, sin_real), __hmul(xq_imag, cos_real));
	nv_bfloat16 xk_out_real = __hsub(__hmul(xk_real, cos_real), __hmul(xk_imag, sin_real));
	nv_bfloat16 xk_out_imag = __hadd(__hmul(xk_real, sin_real), __hmul(xk_imag, cos_real));

	xqOut[row * headDim + col]	   = xq_out_real;
	xqOut[row * headDim + col + 1] = xq_out_imag;
	xkOut[row * headDim + col]	   = xk_out_real;
	xkOut[row * headDim + col + 1] = xk_out_imag;
}

bool precomputeFreqsCis(lyTensor** ppOut, int32_t dim, int32_t end, float theta)
{
	if (!ppOut || dim <= 0 || end <= 0 || theta <= 0)
		return false;

	// Create frequencies tensor
	lyTensor* freqs;
	if (!lyCreateTensor(&freqs, LY_MEMORY_GPU))
		return false;

	int32_t freqsShape[] = {dim / 2};
	if (!lySetTensorShape(freqs, freqsShape, 1) || !lySetTensorData(freqs, NULL, (dim / 2) * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(freqs);
		return false;
	}

	// Compute frequencies
	int freqsBlockSize = 256;
	int freqsGridSize  = (dim / 2 + freqsBlockSize - 1) / freqsBlockSize;

	cudaDeviceSynchronize();
	computeFrequenciesKernel<<<freqsGridSize, freqsBlockSize>>>(freqs->data, dim, end, theta);

	// Create output tensor
	lyTensor* out;
	if (!lyCreateTensor(&out, LY_MEMORY_GPU))
	{
		lyDestroyTensor(freqs);
		return false;
	}

	int32_t outShape[] = {end, dim / 2};
	if (!lySetTensorShape(out, outShape, 2) || !lySetTensorData(out, NULL, end * dim * sizeof(nv_bfloat16)))
	{  // Note: dim not dim/2 here
		lyDestroyTensor(out);
		lyDestroyTensor(freqs);
		return false;
	}

	// Compute rotary embeddings
	dim3 blockSize(256);
	dim3 gridSize((dim / 2 + blockSize.x - 1) / blockSize.x, end);

	cudaDeviceSynchronize();
	computeRotaryKernel<<<gridSize, blockSize>>>(out->data, freqs->data, dim, end);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(out);
		lyDestroyTensor(freqs);
		return false;
	}

	lyDestroyTensor(freqs);

	lyTensorMoveToCPU(out);

	*ppOut = out;
	return true;
}

bool lyApplyRotaryEmbedding(lyTensor** ppXQOut, lyTensor** ppXKOut, lyTensor* pXQ, lyTensor* pXK, lyTensor* pFreqsCis)
{
	if (!ppXQOut || !ppXKOut || !pXQ || !pXK || !pFreqsCis)
	{
		return false;
	}

	lyTensor *pXQOut, *pXKOut;
	if (!lyCreateTensor(&pXQOut, LY_MEMORY_GPU) || !lyCreateTensor(&pXKOut, LY_MEMORY_GPU))
	{
		return false;
	}

	int batchSize	  = pXQ->shape[0];
	int headDim		  = pXQ->shape[1];
	int elementPairs  = batchSize * headDim / 2;  // For complex number handling
	int totalElements = batchSize * headDim;	  // Actual total elements

	// Allocate full size but process as pairs
	if (!lySetTensorShape(pXQOut, pXQ->shape, pXQ->rank) || !lySetTensorData(pXQOut, NULL, totalElements * sizeof(nv_bfloat16)) || !lySetTensorShape(pXKOut, pXK->shape, pXK->rank) || !lySetTensorData(pXKOut, NULL, totalElements * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pXQOut);
		lyDestroyTensor(pXKOut);
		return false;
	}

	int blockSize = 256;
	int gridSize  = (elementPairs + blockSize - 1) / blockSize;	 // Grid size based on pairs

	cudaDeviceSynchronize();
	applyRotaryEmbeddingKernel<<<gridSize, blockSize>>>(pXQOut->data,
														pXKOut->data,
														pXQ->data,
														pXK->data,
														pFreqsCis->data,	  // Real part (cos)
														pFreqsCis->data + 1,  // Imaginary part (sin)
														batchSize,
														headDim);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		lyDestroyTensor(pXQOut);
		lyDestroyTensor(pXKOut);
		return false;
	}

	lyTensorMoveToCPU(pXQOut);
	lyTensorMoveToCPU(pXKOut);

	*ppXQOut = pXQOut;
	*ppXKOut = pXKOut;
	return true;
}