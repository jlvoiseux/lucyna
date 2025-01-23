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
	TEST_ASSERT_TRUE(lyCreateTensor(&pInput, LY_MEMORY_GPU));
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
	TEST_ASSERT_TRUE(lyCreateTensor(&pMask, LY_MEMORY_GPU));
	TEST_ASSERT_TRUE(lySetTensorShape(pMask, maskShape, 2));
	TEST_ASSERT_TRUE(lySetTensorData(pMask, NULL, 16 * sizeof(nv_bfloat16)));
	TEST_ASSERT_TRUE(lyTensorMakeTriangularMask(pMask));

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyAttentionForward(&pOutput, pAttention, pInput, 0, pFreqsCis, pMask));

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(4, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(pModel->args.dim, pOutput->shape[1]);

	TEST_ASSERT_EQUAL_INT32(128, pAttention->cacheK->shape[0]);
	TEST_ASSERT_EQUAL_INT32(pAttention->nKVHeads, pAttention->cacheK->shape[1]);
	TEST_ASSERT_EQUAL_INT32(pAttention->headDim, pAttention->cacheK->shape[2]);

	lyDestroyTensor(pOutput);
	lyDestroyTensor(pMask);
	lyDestroyTensor(pFreqsCis);
	lyDestroyTensor(pInput);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_AttentionForward);
	return UNITY_END();
}