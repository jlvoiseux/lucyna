#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"

#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510582097494459
#endif

void lyRopePrecomputeFreqsCis(lyTensorDouble** ppOut, int32_t dim, int32_t end, double theta)
{
	lyTensor* freqs;
	int32_t	  freqsShape[] = {dim / 2};
	lyTensorCreate(&freqs, freqsShape, 1, NULL, NULL);

	for (int32_t i = 0; i < dim / 2; i++)
	{
		float val	   = (float)(2 * i);
		float freq	   = (float)(1.0 / pow(theta, (double)(val / dim)));
		freqs->data[i] = __float2bfloat16_rz(freq);
	}
	lyTensorPrint(freqs);

	lyTensor* t;
	int32_t	  tShape[] = {end};
	lyTensorCreate(&t, tShape, 1, NULL, NULL);
	for (int32_t i = 0; i < end; i++)
	{
		t->data[i] = __float2bfloat16_rz((float)i);
	}
	lyTensorPrint(t);

	for (int32_t i = 0; i < dim / 2; i++)
	{
		float freq = __bfloat162float(freqs->data[i]);

		float wavelen = 2.0 * M_PI / (double)freq;

		const float scaleFactor		= 8.0f;
		const float lowFreqFactor	= 1.0f;
		const float highFreqFactor	= 4.0f;
		const float oldContextLen	= 8192.0f;
		const float lowFreqWavelen	= oldContextLen / lowFreqFactor;
		const float highFreqWavelen = oldContextLen / highFreqFactor;

		if (wavelen < highFreqWavelen)
		{
			freq = freq;
		}
		else if (wavelen > lowFreqWavelen)
		{
			freq = freq / scaleFactor;
		}
		else
		{
			float smooth = (oldContextLen / wavelen - lowFreqFactor) / (highFreqFactor - lowFreqFactor);
			freq		 = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
		}

		freqs->data[i] = __float2bfloat16_rz(freq);
	}

	lyTensorPrint(freqs);

	lyTensor* outerProduct;
	lyTensorOuter(&outerProduct, t, freqs);
	lyTensorPrint(outerProduct);

	lyTensorDouble* out;
	int32_t			outShape[] = {end, dim};
	lyTensorDoubleCreate(&out, outShape, 2, NULL, NULL);

	for (int32_t i = 0; i < end; i++)
	{
		for (int32_t j = 0; j < dim / 2; j++)
		{
			float	angle		   = __bfloat162float(outerProduct->data[i * (dim / 2) + j]);
			int32_t baseIdx		   = i * dim + 2 * j;
			out->data[baseIdx]	   = cos(angle);
			out->data[baseIdx + 1] = sin(angle);
		}
	}

	lyTensorDoublePrint(out);

	lyTensorDestroy(freqs);
	lyTensorDestroy(t);
	lyTensorDestroy(outerProduct);

	*ppOut = out;
}

void lyRopeApplyEmbeddings(lyTensor** ppXQOut, lyTensor** ppXKOut, lyTensor* pXQ, lyTensor* pXK, lyTensorDouble* pFreqsCis)
{
	lyTensor *pXQOut, *pXKOut;
	lyTensorCreate(&pXQOut, pXQ->shape, pXQ->rank, NULL, NULL);
	lyTensorCreate(&pXKOut, pXK->shape, pXK->rank, NULL, NULL);

	int seqLen	  = pXQ->shape[0];
	int numHeadsQ = pXQ->shape[1];
	int numHeadsK = pXK->shape[1];
	int headDim	  = pXQ->shape[2];

	for (int pos = 0; pos < seqLen; pos++)
	{
		for (int head = 0; head < numHeadsQ; head++)
		{
			for (int dim = 0; dim < headDim / 2; dim++)
			{
				int	   inIdxQ  = pos * numHeadsQ * headDim + head * headDim + 2 * dim;
				double xq_real = __bfloat162float(pXQ->data[inIdxQ]);
				double xq_imag = __bfloat162float(pXQ->data[inIdxQ + 1]);

				int	   freqsIdx = pos * headDim + 2 * dim;
				double cos_val	= pFreqsCis->data[freqsIdx];
				double sin_val	= pFreqsCis->data[freqsIdx + 1];

				pXQOut->data[inIdxQ]	 = __float2bfloat16_rz(xq_real * cos_val - xq_imag * sin_val);
				pXQOut->data[inIdxQ + 1] = __float2bfloat16_rz(xq_real * sin_val + xq_imag * cos_val);
			}
		}

		for (int head = 0; head < numHeadsK; head++)
		{
			for (int dim = 0; dim < headDim / 2; dim++)
			{
				int	   inIdxK  = pos * numHeadsK * headDim + head * headDim + 2 * dim;
				double xk_real = __bfloat162float(pXK->data[inIdxK]);
				double xk_imag = __bfloat162float(pXK->data[inIdxK + 1]);

				int	   freqsIdx = pos * headDim + 2 * dim;
				double cos_val	= pFreqsCis->data[freqsIdx];
				double sin_val	= pFreqsCis->data[freqsIdx + 1];

				pXKOut->data[inIdxK]	 = __float2bfloat16_rz(xk_real * cos_val - xk_imag * sin_val);
				pXKOut->data[inIdxK + 1] = __float2bfloat16_rz(xk_real * sin_val + xk_imag * cos_val);
			}
		}
	}

	*ppXQOut = pXQOut;
	*ppXKOut = pXKOut;
}