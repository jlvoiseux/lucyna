#include "lyTransformer.h"

#include "lyModelLoader.h"
#include "lyOpenCL.h"
#include "lyRotaryPosEmbeddings.h"
#include "lyTensor.h"
#include "lyTensorMath.h"
#include "lyUtil.h"
#include "unity.h"

#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static lyModel*			pModel		   = NULL;
static lyOpenCLContext* pOpenCLContext = NULL;

void setUp(void)
{
	lyOpenCLInit(&pOpenCLContext);
	lyModelLoaderLoadModel(&pModel, "../model-tuned");
}

void tearDown(void)
{
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

bool logCallback(const char* format, ...)
{
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	return true;
}

void test_TransformerForward(void)
{
	if (!pModel)
	{
		TEST_IGNORE_MESSAGE("Model not found, skipping test");
		return;
	}

	if (!pOpenCLContext || !pOpenCLContext->initialized)
	{
		TEST_IGNORE_MESSAGE("OpenCL context not initialized, skipping test");
		return;
	}

	const int32_t SEQ_LENGTH	 = 20;
	const int32_t promptTokens[] = {128000, 128006, 882, 128007, 271, 3923, 374, 701, 836, 30, 128009, 128006, 78191, 128007, 271};
	const int32_t promptLength	 = sizeof(promptTokens) / sizeof(promptTokens[0]);

	lyTransformer* pTransformer;
	lyTransformerCreate(&pTransformer, pModel, pOpenCLContext);

	int32_t	  prevPos = 0;
	lyTensor* pLogits;
	lyTensor* pLastLogits;
	int32_t*  generatedTokens = (int32_t*)malloc(SEQ_LENGTH * sizeof(int32_t));

	memcpy(generatedTokens, promptTokens, promptLength * sizeof(int32_t));

	for (int32_t curPos = promptLength; curPos < SEQ_LENGTH; curPos++)
	{
		lyTransformerForward(&pLogits, pTransformer, generatedTokens, curPos, prevPos);
		lyTensorSlice(&pLastLogits, pLogits, pLogits->shape[0] - 1, pLogits->shape[0]);
		lyTensorDestroy(pLogits);

		int32_t nextToken;
		lyTensorArgmax(&nextToken, pLastLogits, pOpenCLContext);
		lyTensorDestroy(pLastLogits);

		generatedTokens[curPos] = nextToken;
		prevPos					= curPos;
	}

	TEST_ASSERT_GREATER_OR_EQUAL_INT32(0, generatedTokens[promptLength]);
	TEST_ASSERT_LESS_THAN_INT32(pModel->args.vocabSize, generatedTokens[promptLength]);

	free(generatedTokens);
	lyTransformerDestroy(pTransformer);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_TransformerForward);
	return UNITY_END();
}