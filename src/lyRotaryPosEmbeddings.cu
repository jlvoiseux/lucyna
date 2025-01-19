#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"

#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static bool applyScaling(lyTensor* pFreqs)
{
	if (!pFreqs)
	{
		return false;
	}

	const float scaleFactor		= 8.0f;
	const float lowFreqFactor	= 1.0f;
	const float highFreqFactor	= 4.0f;
	const float oldContextLen	= 8192.0f;	// original llama3 length
	const float lowFreqWavelen	= oldContextLen / lowFreqFactor;
	const float highFreqWavelen = oldContextLen / highFreqFactor;

	int size = pFreqs->shape[0];

	for (int i = 0; i < size; i++)
	{
		float freq;
		if (!lyTensorGetItemAsFloat32(&freq, pFreqs, i))
		{
			return false;
		}

		float wavelen = 2.0f * M_PI / freq;
		float newFreq;

		if (wavelen < highFreqWavelen)
		{
			newFreq = freq;
		}
		else if (wavelen > lowFreqWavelen)
		{
			newFreq = freq / scaleFactor;
		}
		else
		{
			float smooth = (oldContextLen / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
			newFreq		 = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
		}

		if (!lyTensorSetItemFromFloat32(pFreqs, i, newFreq))
		{
			return false;
		}
	}

	return true;
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
	{
		return false;
	}

	lyTensor* freqs;
	int32_t	  freqsShape[] = {dim / 2};
	if (!lyCreateTensor(&freqs))
	{
		return false;
	}
	if (!lySetTensorShape(freqs, freqsShape, 1) || !lySetTensorData(freqs, NULL, (dim / 2) * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(freqs);
		return false;
	}

	float dimFloat = (float)dim;
	for (int i = 0; i < dim / 2; i++)
	{
		float val = (float)(1.0 / pow(theta, (2.0f * i) / dimFloat));
		if (!lyTensorSetItemFromFloat32(freqs, i, val))
		{
			lyDestroyTensor(freqs);
			return false;
		}
	}

	if (!applyScaling(freqs))
	{
		lyDestroyTensor(freqs);
		return false;
	}
	for (int i = 0; i < dim / 2; i++)
	{
		float val;
		lyTensorGetItemAsFloat32(&val, freqs, i);
	}

	lyTensor* t;
	int32_t	  tShape[] = {end};
	if (!lyCreateTensor(&t))
	{
		lyDestroyTensor(freqs);
		return false;
	}
	if (!lySetTensorShape(t, tShape, 1) || !lySetTensorData(t, NULL, end * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(t);
		lyDestroyTensor(freqs);
		return false;
	}

	for (int i = 0; i < end; i++)
	{
		if (!lyTensorSetItemFromFloat32(t, i, (float)i))
		{
			lyDestroyTensor(t);
			lyDestroyTensor(freqs);
			return false;
		}
	}

	lyTensor* freqsOuter;
	if (!lyTensorOuter(&freqsOuter, t, freqs))
	{
		lyDestroyTensor(t);
		lyDestroyTensor(freqs);
		return false;
	}

	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			float val;
			lyTensorGetItemAsFloat32(&val, freqsOuter, i * dim / 2 + j);
		}
	}

	int32_t	  outShape[] = {end, dim / 2};
	lyTensor* out;
	if (!lyCreateTensor(&out))
	{
		lyDestroyTensor(freqsOuter);
		lyDestroyTensor(t);
		lyDestroyTensor(freqs);
		return false;
	}
	if (!lySetTensorShape(out, outShape, 2) || !lySetTensorData(out, NULL, end * dim * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(out);
		lyDestroyTensor(freqsOuter);
		lyDestroyTensor(t);
		lyDestroyTensor(freqs);
		return false;
	}

	for (int i = 0; i < end; i++)
	{
		for (int j = 0; j < dim / 2; j++)
		{
			float angle;
			if (!lyTensorGetItemAsFloat32(&angle, freqsOuter, i * dim / 2 + j))
			{
				lyDestroyTensor(out);
				lyDestroyTensor(freqsOuter);
				lyDestroyTensor(t);
				lyDestroyTensor(freqs);
				return false;
			}

			float cos_val = cosf(angle);
			float sin_val = sinf(angle);

			if (!lyTensorSetComplexItem(out, i, j, cos_val, sin_val))
			{
				lyDestroyTensor(out);
				lyDestroyTensor(freqsOuter);
				lyDestroyTensor(t);
				lyDestroyTensor(freqs);
				return false;
			}
		}

		cudaDeviceSynchronize();
	}

	lyDestroyTensor(freqsOuter);
	lyDestroyTensor(t);
	lyDestroyTensor(freqs);

	*ppOut = out;
	return true;
}

bool lyApplyRotaryEmbedding(lyTensor** ppXQOut, lyTensor** ppXKOut, const lyTensor* pXQ, const lyTensor* pXK, const lyTensor* pFreqsCis)
{
	if (!ppXQOut || !ppXKOut || !pXQ || !pXK || !pFreqsCis)
	{
		return false;
	}

	lyTensor *pXQOut, *pXKOut;
	if (!lyCreateTensor(&pXQOut) || !lyCreateTensor(&pXKOut))
	{
		return false;
	}

	int batchSize	  = pXQ->shape[0];
	int headDim		  = pXQ->shape[1];
	int totalElements = batchSize * headDim / 2;

	if (!lySetTensorShape(pXQOut, pXQ->shape, pXQ->rank) || !lySetTensorData(pXQOut, NULL, totalElements * sizeof(nv_bfloat16)) || !lySetTensorShape(pXKOut, pXK->shape, pXK->rank) || !lySetTensorData(pXKOut, NULL, totalElements * sizeof(nv_bfloat16)))
	{
		lyDestroyTensor(pXQOut);
		lyDestroyTensor(pXKOut);
		return false;
	}

	int blockSize = 256;
	int gridSize  = (totalElements + blockSize - 1) / blockSize;

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

	*ppXQOut = pXQOut;
	*ppXKOut = pXKOut;
	return true;
}