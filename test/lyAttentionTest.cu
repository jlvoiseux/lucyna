#include "lyAttention.h"
#include "lyModel.h"
#include "lyModelLoader.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensorMath.h"
#include "unity.h"

static lyModel*		pModel	   = NULL;
static lyAttention* pAttention = NULL;

void setUp(void)
{
	TEST_ASSERT_TRUE(lyModelLoaderLoadModel(&pModel, "../model-tuned", true, true));
	TEST_ASSERT_TRUE(lyAttentionCreate(&pAttention, pModel, 0));
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
}

void test_AttentionForward(void)
{
	int32_t	  inputShape[] = {4, pModel->args.dim};
	lyTensor* pInput;
	lyTensorCreate(&pInput, inputShape, 2, NULL, NULL);

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < pModel->args.dim; j++)
		{
			float val = (float)(i * pModel->args.dim + j) / (float)(4 * pModel->args.dim);
			TEST_ASSERT_TRUE(lyTensorSetItemFromFloat32(pInput, i * pModel->args.dim + j, val));
		}
	}

	lyTensor* pFreqsCis;
	TEST_ASSERT_TRUE(lyRopePrecomputeFreqsCis(&pFreqsCis, pModel->args.dim, 4096, 10000.0f));

	int32_t	  maskShape[] = {4, 4};
	lyTensor* pMask;
	lyTensorCreate(&pMask, maskShape, 2, NULL, NULL);
	TEST_ASSERT_TRUE(lyTensorMakeTriangularMask(pMask));

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyAttentionForward(&pOutput, pAttention, pInput, 0, pFreqsCis, pMask));

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(4, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(pModel->args.dim, pOutput->shape[1]);

	TEST_ASSERT_EQUAL_INT32(pModel->args.maxSequenceLength, pAttention->cacheK->shape[0]);
	TEST_ASSERT_EQUAL_INT32(pModel->args.nKVHeads, pAttention->cacheK->shape[1]);
	TEST_ASSERT_EQUAL_INT32(pModel->args.headDim, pAttention->cacheK->shape[2]);

	lyTensorDestroy(pOutput);
	lyTensorDestroy(pMask);
	lyTensorDestroy(pFreqsCis);
	lyTensorDestroy(pInput);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_AttentionForward);
	return UNITY_END();
}