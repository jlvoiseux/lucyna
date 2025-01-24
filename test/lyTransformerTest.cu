#include "lyModelLoader.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensor.h"
#include "lyTensorMath.h"
#include "lyTransformer.h"
#include "lyUtil.h"
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

bool logCallback(const char* format, ...)
{
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	return true;
}

static bool verifyPrepare(lyTensor* pInputTensor, lyTensor* pFreqsCis, lyTensor* pMask)
{
	// Verify input tensor shape
	TEST_ASSERT_EQUAL_INT32(2, pInputTensor->rank);
	TEST_ASSERT_EQUAL_INT32(15, pInputTensor->shape[0]);
	TEST_ASSERT_EQUAL_INT32(4096, pInputTensor->shape[1]);

	// Verify freqsCis shape
	TEST_ASSERT_EQUAL_INT32(2, pFreqsCis->rank);
	TEST_ASSERT_EQUAL_INT32(15, pFreqsCis->shape[0]);
	TEST_ASSERT_EQUAL_INT32(64, pFreqsCis->shape[1]);

	// Verify mask shape
	TEST_ASSERT_EQUAL_INT32(2, pMask->rank);
	TEST_ASSERT_EQUAL_INT32(15, pMask->shape[0]);
	TEST_ASSERT_EQUAL_INT32(15, pMask->shape[1]);

	return true;
}

static bool verifyOutput(lyTensor* pOutput)
{
	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(1, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(128256, pOutput->shape[1]);

	return true;
}

void test_TransformerForward(void)
{
	if (!pModel)
	{
		TEST_IGNORE_MESSAGE("Model not found, skipping test");
		return;
	}

	const int32_t SEQ_LENGTH = 20;

	int32_t	  shape[] = {SEQ_LENGTH};
	lyTensor* pTokens;
	lyCreateTensor(&pTokens, shape, 1, NULL, NULL);

	const int32_t padToken = -1;  // Get from model
	for (int32_t i = 0; i < SEQ_LENGTH; i++)
	{
		TEST_ASSERT_TRUE(lyTensorSetItem(pTokens, &i, padToken));
	}

	const int32_t promptTokens[] = {128000, 128006, 882, 128007, 271, 3923, 374, 701, 836, 30, 128009, 128006, 78191, 128007, 271};
	const int32_t promptLength	 = sizeof(promptTokens) / sizeof(promptTokens[0]);

	for (int32_t i = 0; i < promptLength; i++)
	{
		TEST_ASSERT_TRUE(lyTensorSetItem(pTokens, &i, promptTokens[i]));
	}

	lyTransformer* pTransformer;
	TEST_ASSERT_TRUE(lyCreateTransformer(&pTransformer, pModel));

	int32_t	  prevPos = 0;
	lyTensor *pInputTokens, *pLogits, *pNextToken;

	for (int32_t curPos = promptLength; curPos < SEQ_LENGTH; curPos++)
	{
		lyTensorSlice(&pInputTokens, pTokens, prevPos, curPos);
		TEST_ASSERT_TRUE(lyTransformerForward(&pLogits, pTransformer, pInputTokens, prevPos));
		lyTensorSlice(&pNextToken, pLogits, pLogits->shape[0] - 1, pLogits->shape[0]);
		lyTensor* pArgmax;
		TEST_ASSERT_TRUE(lyTensorArgmax(&pArgmax, pNextToken, pNextToken->rank - 1));
		int32_t nextTokenId;
		TEST_ASSERT_TRUE(lyTensorGetItem(&nextTokenId, pArgmax, 0));
		TEST_ASSERT_TRUE(lyTensorSetItem(pTokens, &curPos, nextTokenId));
		lyDestroyTensor(pInputTokens);
		lyDestroyTensor(pLogits);
		lyDestroyTensor(pNextToken);
		lyDestroyTensor(pArgmax);

		prevPos = curPos;
	}

	lyDestroyTransformer(pTransformer);
	lyDestroyTensor(pTokens);
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
	// RUN_TEST(test_TransformerForward);
	RUN_TEST(test_PrecomputeFreqsCis);
	return UNITY_END();
}