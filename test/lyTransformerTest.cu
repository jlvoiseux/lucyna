#include "lyModelLoader.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensor.h"
#include "lyTransformer.h"
#include "unity.h"

#define M_PI 3.14159265358979323846f

static lyModel* pModel = NULL;

void setUp(void)
{
	lyLoadModel(&pModel, "../model-tuned", true, true);
}

void tearDown(void)
{
	if (pModel)
	{
		lyDestroyModel(pModel);
		pModel = NULL;
	}
}

void test_TransformerForward(void)
{
	lyTransformer* pTransformer;
	lyCreateTransformer(&pTransformer, pModel);

	int32_t	  tokenShape[] = {4};
	lyTensor* pTokens;
	TEST_ASSERT_TRUE(lyCreateTensor(&pTokens));
	TEST_ASSERT_TRUE(lySetTensorShape(pTokens, tokenShape, 1));
	TEST_ASSERT_TRUE(lySetTensorData(pTokens, NULL, 4 * sizeof(nv_bfloat16)));

	int32_t testTokens[] = {1, 2, 3, 4};
	for (int i = 0; i < 4; i++)
	{
		TEST_ASSERT_TRUE(lyTensorSetItem(pTokens, &i, testTokens[i]));
	}

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTransformerForward(&pOutput, pTransformer, pTokens, 0));

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(4, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(pTransformer->vocabSize, pOutput->shape[1]);

	lyDestroyTensor(pOutput);
	lyDestroyTensor(pTokens);

	if (pTransformer)
	{
		lyDestroyTransformer(pTransformer);
		pTransformer = NULL;
	}
}

void test_PrecomputeFreqsCis(void)
{
	float	  expected[] = {57.29578f, 46.67413f, 38.02155f, 30.97301f, 25.23115f, 20.55373f, 16.74342f, 13.63948f, 11.11096f, 9.05118f};
	lyTensor* freqsCis;
	TEST_ASSERT_TRUE(precomputeFreqsCis(&freqsCis, 128, 4096, 500000.0));

	TEST_ASSERT_EQUAL_INT32(2, freqsCis->rank);
	TEST_ASSERT_EQUAL_INT32(4096, freqsCis->shape[0]);
	TEST_ASSERT_EQUAL_INT32(64, freqsCis->shape[1]);

	for (int i = 0; i < 10; i++)
	{
		float real, imag;
		if (lyTensorGetComplexItem(&real, &imag, freqsCis, 1, i))
		{
			float angle = atan2(imag, real) * (180.0 / M_PI);
			TEST_ASSERT_FLOAT_WITHIN(0.5f, expected[i], angle);
		}
	}

	lyDestroyTensor(freqsCis);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_TransformerForward);
	// RUN_TEST(test_PrecomputeFreqsCis);
	return UNITY_END();
}