#include "lyAttention.h"

#include "lyModel.h"
#include "lyModelLoader.h"
#include "lyOpenCL.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"
#include "unity.h"

static lyModel*			pModel		   = NULL;
static lyAttention*		pAttention	   = NULL;
static lyOpenCLContext* pOpenCLContext = NULL;

void setUp(void)
{
	lyOpenCLInit(&pOpenCLContext);
	lyModelLoaderLoadModel(&pModel, "../model-tuned");
	lyAttentionCreate(&pAttention, pModel, 0, pOpenCLContext);
}

void tearDown(void)
{
	if (pAttention)
	{
		lyAttentionDestroy(pAttention);
		pAttention = NULL;
	}
	if (pModel)
	{
		lyModelLoaderDestroyModel(pModel);
		pModel = NULL;
	}
	if (pOpenCLContext)
	{
		lyOpenCLDestroy(pOpenCLContext);
		pOpenCLContext = NULL;
	}
}

void test_AttentionForward(void)
{
	if (!pOpenCLContext || !pOpenCLContext->initialized)
	{
		TEST_IGNORE_MESSAGE("OpenCL context not initialized, skipping test");
		return;
	}

	int32_t	  inputShape[] = {4, pModel->args.dim};
	lyTensor* pInput;
	lyTensorCreate(&pInput, inputShape, 2, NULL, NULL);

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < pModel->args.dim; j++)
		{
			float	val	  = (float)(i * pModel->args.dim + j) / (float)(4 * pModel->args.dim);
			int32_t loc[] = {i, j};
			lyTensorSetItem(pInput, loc, val);
		}
	}

	lyTensorDouble* pFreqsCis;
	lyRopePrecomputeFreqsCis(&pFreqsCis, pModel->args.dim, 4096, 10000.0f, pOpenCLContext);

	int32_t	  maskShape[] = {4, 4};
	lyTensor* pMask;
	lyTensorCreate(&pMask, maskShape, 2, NULL, NULL);
	lyTensorMakeTriangularMask(pMask, pOpenCLContext);

	lyTensor* pOutput;
	lyAttentionForward(&pOutput, pAttention, pInput, 0, pFreqsCis, pMask);

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(4, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(pModel->args.dim, pOutput->shape[1]);

	TEST_ASSERT_EQUAL_INT32(pModel->args.maxSequenceLength, pAttention->cacheK->shape[0]);
	TEST_ASSERT_EQUAL_INT32(pModel->args.nKVHeads, pAttention->cacheK->shape[1]);
	TEST_ASSERT_EQUAL_INT32(pModel->args.headDim, pAttention->cacheK->shape[2]);

	lyTensorDestroy(pOutput);
	lyTensorDestroy(pMask);
	lyTensorDoubleDestroy(pFreqsCis);
	lyTensorDestroy(pInput);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_AttentionForward);
	return UNITY_END();
}