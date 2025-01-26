#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"

#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

void lyRopePrecomputeFreqsCis(lyTensor** ppOut, int32_t dim, int32_t end, float theta)
{
	lyTensor* freqs;
	int32_t	  freqsShape[] = {dim / 2};
	lyTensorCreate(&freqs, freqsShape, 1, NULL, NULL);

	float dimFloat = (float)dim;
	for (int idx = 0; idx < dim / 2; idx++)
	{
		float freq	  = 1.0f / powf(theta, (2.0f * idx) / dimFloat);
		float wavelen = 2.0f * M_PI / freq;

		const float scaleFactor		= 8.0f;
		const float lowFreqFactor	= 1.0f;
		const float highFreqFactor	= 4.0f;
		const float oldContextLen	= 8192.0f;
		const float lowFreqWavelen	= oldContextLen / lowFreqFactor;
		const float highFreqWavelen = oldContextLen / highFreqFactor;

		if (wavelen < highFreqWavelen)
			freq = freq;
		else if (wavelen > lowFreqWavelen)
			freq = freq / scaleFactor;
		else
		{
			float smooth = (oldContextLen / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
			freq		 = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
		}

		freqs->data[idx] = __float2bfloat16(freq);
	}

	lyTensor* out;
	int32_t	  outShape[] = {end, dim};
	lyTensorCreate(&out, outShape, 2, NULL, NULL);

	for (int row = 0; row < end; row++)
	{
		for (int col = 0; col < dim / 2; col++)
		{
			float freq	= __bfloat162float(freqs->data[col]);
			float angle = row * freq;

			int baseIdx			   = row * dim + 2 * col;
			out->data[baseIdx]	   = __float2bfloat16(cosf(angle));
			out->data[baseIdx + 1] = __float2bfloat16(sinf(angle));
		}
	}

	lyTensorDestroy(freqs);
	*ppOut = out;
}

void lyRopeApplyEmbeddings(lyTensor** ppXQOut, lyTensor** ppXKOut, lyTensor* pXQ, lyTensor* pXK, lyTensor* pFreqsCis)
{
	lyTensor *pXQOut, *pXKOut;
	lyTensorCreate(&pXQOut, pXQ->shape, pXQ->rank, NULL, NULL);
	lyTensorCreate(&pXKOut, pXK->shape, pXK->rank, NULL, NULL);

	int batchSize  = pXQ->shape[0];
	int headDim	   = pXQ->shape[1];
	int totalPairs = batchSize * headDim / 2;

	for (int idx = 0; idx < totalPairs; idx++)
	{
		int row = idx / (headDim / 2);
		int col = (idx % (headDim / 2)) * 2;

		nv_bfloat16 xq_real = pXQ->data[row * headDim + col];
		nv_bfloat16 xq_imag = pXQ->data[row * headDim + col + 1];
		nv_bfloat16 xk_real = pXK->data[row * headDim + col];
		nv_bfloat16 xk_imag = pXK->data[row * headDim + col + 1];

		nv_bfloat16 cos_real = pFreqsCis->data[col / 2];
		nv_bfloat16 sin_real = pFreqsCis->data[col / 2 + 1];

		nv_bfloat16 xq_out_real = __hsub(__hmul(xq_real, cos_real), __hmul(xq_imag, sin_real));
		nv_bfloat16 xq_out_imag = __hadd(__hmul(xq_real, sin_real), __hmul(xq_imag, cos_real));
		nv_bfloat16 xk_out_real = __hsub(__hmul(xk_real, cos_real), __hmul(xk_imag, sin_real));
		nv_bfloat16 xk_out_imag = __hadd(__hmul(xk_real, sin_real), __hmul(xk_imag, cos_real));

		pXQOut->data[row * headDim + col]	  = xq_out_real;
		pXQOut->data[row * headDim + col + 1] = xq_out_imag;
		pXKOut->data[row * headDim + col]	  = xk_out_real;
		pXKOut->data[row * headDim + col + 1] = xk_out_imag;
	}

	*ppXQOut = pXQOut;
	*ppXKOut = pXKOut;
}