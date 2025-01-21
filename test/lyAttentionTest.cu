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
	TEST_ASSERT_TRUE(lyLoadModel(&pModel, "../model-tuned", true, true));
	TEST_ASSERT_TRUE(lyCreateAttention(&pAttention, pModel, 0));
}

void tearDown(void)
{
	if (pAttention)
	{
		lyDestroyAttention(pAttention);
		pAttention = NULL;
	}
	if (pModel)
	{
		lyDestroyModel(pModel);
		pModel = NULL;
	}
}

void test_AttentionForward(void)
{
	int32_t	  inputShape[] = {4, pModel->args.dim};
	lyTensor* pInput;
	TEST_ASSERT_TRUE(lyCreateTensor(&pInput));
	TEST_ASSERT_TRUE(lySetTensorShape(pInput, inputShape, 2));
	TEST_ASSERT_TRUE(lySetTensorData(pInput, NULL, 4 * pModel->args.dim * sizeof(nv_bfloat16)));

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < pModel->args.dim; j++)
		{
			float val = (float)(i * pModel->args.dim + j) / (float)(4 * pModel->args.dim);
			TEST_ASSERT_TRUE(lyTensorSetItemFromFloat32(pInput, i * pModel->args.dim + j, val));
		}
	}

	lyTensor* pFreqsCis;
	TEST_ASSERT_TRUE(precomputeFreqsCis(&pFreqsCis, pModel->args.dim, 4096, 10000.0f));

	int32_t	  maskShape[] = {4, 4};
	lyTensor* pMask;
	TEST_ASSERT_TRUE(lyCreateTensor(&pMask));
	TEST_ASSERT_TRUE(lySetTensorShape(pMask, maskShape, 2));
	TEST_ASSERT_TRUE(lySetTensorData(pMask, NULL, 16 * sizeof(nv_bfloat16)));
	TEST_ASSERT_TRUE(lyTensorMakeTriangularMask(pMask));

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyAttentionForward(&pOutput, pAttention, pInput, 0, pFreqsCis, pMask));

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(4, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(pModel->args.dim, pOutput->shape[1]);

	TEST_ASSERT_EQUAL_INT32(4, pAttention->cacheK->shape[0]);
	TEST_ASSERT_EQUAL_INT32(pAttention->nKVHeads, pAttention->cacheK->shape[1]);
	TEST_ASSERT_EQUAL_INT32(pAttention->headDim, pAttention->cacheK->shape[2]);

	lyDestroyTensor(pOutput);
	lyDestroyTensor(pMask);
	lyDestroyTensor(pFreqsCis);
	lyDestroyTensor(pInput);
}

void test_KVCache(void)
{
	int32_t	  inputShape[] = {2, pModel->args.dim};
	lyTensor* pK;
	lyTensor* pV;
	TEST_ASSERT_TRUE(lyCreateTensor(&pK));
	TEST_ASSERT_TRUE(lyCreateTensor(&pV));
	TEST_ASSERT_TRUE(lySetTensorShape(pK, inputShape, 2));
	TEST_ASSERT_TRUE(lySetTensorShape(pV, inputShape, 2));
	TEST_ASSERT_TRUE(lySetTensorData(pK, NULL, 2 * pModel->args.dim * sizeof(nv_bfloat16)));
	TEST_ASSERT_TRUE(lySetTensorData(pV, NULL, 2 * pModel->args.dim * sizeof(nv_bfloat16)));

	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < pModel->args.dim; j++)
		{
			float val = (float)(i + j) / (float)(2 * pModel->args.dim);
			TEST_ASSERT_TRUE(lyTensorSetItemFromFloat32(pK, i * pModel->args.dim + j, val));
			TEST_ASSERT_TRUE(lyTensorSetItemFromFloat32(pV, i * pModel->args.dim + j, val + 1.0f));
		}
	}

	TEST_ASSERT_TRUE(lyUpdateKVCache(pAttention, pK, pV, 0));

	float kVal, vVal;
	TEST_ASSERT_TRUE(lyTensorGetItemAsFloat32(&kVal, pAttention->cacheK, 0));
	TEST_ASSERT_TRUE(lyTensorGetItemAsFloat32(&vVal, pAttention->cacheV, 0));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, kVal);
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, vVal);

	lyDestroyTensor(pK);
	lyDestroyTensor(pV);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_AttentionForward);
	// RUN_TEST(test_KVCache);
	return UNITY_END();
}