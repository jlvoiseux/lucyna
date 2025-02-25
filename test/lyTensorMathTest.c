#include "lyTensorMath.h"

#include "lyTensor.h"
#include "unity.h"

#include <lyOpenCL.h>
#include <math.h>
#include <stdlib.h>

static lyTensor*		pTensorA = NULL;
static lyTensor*		pTensorB = NULL;
static lyOpenCLContext* pContext = NULL;

void setUp(void)
{
	lyOpenCLInit(&pContext);
}

void tearDown(void)
{
	if (pTensorA)
	{
		lyTensorDestroy(pTensorA);
		pTensorA = NULL;
	}
	if (pTensorB)
	{
		lyTensorDestroy(pTensorB);
		pTensorB = NULL;
	}
	if (pContext)
	{
		lyOpenCLDestroy(pContext);
		pContext = NULL;
	}
}

void test_TensorMatMul2D(void)
{
	int32_t shapeA[] = {2, 3};
	int32_t shapeB[] = {3, 2};

	lyBfloat16 dataA[6];
	for (int i = 0; i < 6; i++)
	{
		dataA[i] = lyFloat32ToBfloat16((float)(i + 1));
	}

	lyBfloat16 dataB[6];
	for (int i = 0; i < 6; i++)
	{
		dataB[i] = lyFloat32ToBfloat16((float)(i + 1));
	}

	lyTensorCreate(&pTensorA, shapeA, 2, dataA, NULL);
	lyTensorCreate(&pTensorB, shapeB, 2, dataB, NULL);

	lyTensor* pOutput;
	lyTensorMatMul(&pOutput, pTensorA, pTensorB, pContext);

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);

	float expected[] = {22.0f, 28.0f, 49.0f, 64.0f};
	for (int i = 0; i < 4; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], lyBfloat16ToFloat32(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_TensorMatMul3D(void)
{
	// [2, 3, 4] x [2, 4, 3] -> [2, 3, 3]
	int32_t shapeA[] = {2, 3, 4};
	int32_t shapeB[] = {2, 4, 3};

	lyBfloat16* dataA = (lyBfloat16*)malloc(24 * sizeof(lyBfloat16));
	lyBfloat16* dataB = (lyBfloat16*)malloc(24 * sizeof(lyBfloat16));

	for (int i = 0; i < 24; i++)
	{
		dataA[i] = lyFloat32ToBfloat16((float)(i + 1));
		dataB[i] = lyFloat32ToBfloat16((float)(i + 1));
	}

	lyTensor *pA, *pB;
	lyTensorCreate(&pA, shapeA, 3, dataA, NULL);
	lyTensorCreate(&pB, shapeB, 3, dataB, NULL);

	lyTensor* pOutput;
	lyTensorMatMul(&pOutput, pA, pB, pContext);

	TEST_ASSERT_EQUAL_INT32(3, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[1]);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[2]);

	float expected_first[] = {70, 80, 90, 158, 184, 210, 246, 288, 330};

	for (int i = 0; i < 9; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.1f, expected_first[i], lyBfloat16ToFloat32(pOutput->data[i]));
	}

	free(dataA);
	free(dataB);
	lyTensorDestroy(pA);
	lyTensorDestroy(pB);
	lyTensorDestroy(pOutput);
}

void test_TensorScaleAndAdd2D(void)
{
	int32_t shape[] = {2, 3};

	lyBfloat16 dataA[6];
	for (int i = 0; i < 6; i++)
	{
		dataA[i] = lyFloat32ToBfloat16((float)(i + 1));
	}

	lyBfloat16 dataB[6];
	for (int i = 0; i < 6; i++)
	{
		dataB[i] = lyFloat32ToBfloat16((float)(i + 1) * 0.5f);
	}

	lyTensorCreate(&pTensorA, shape, 2, dataA, NULL);
	lyTensorCreate(&pTensorB, shape, 2, dataB, NULL);

	lyTensor*  pOutput;
	lyBfloat16 alpha = lyFloat32ToBfloat16(2.0f);
	lyBfloat16 beta	 = lyFloat32ToBfloat16(-1.0f);
	lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, alpha, beta, pContext);

	float expected[] = {1.5f, 3.0f, 4.5f, 6.0f, 7.5f, 9.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], lyBfloat16ToFloat32(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_TensorElementwiseMul(void)
{
	int32_t shape[] = {2, 3};

	lyBfloat16 dataA[6];
	for (int i = 0; i < 6; i++)
	{
		dataA[i] = lyFloat32ToBfloat16((float)(i + 1));
	}

	lyBfloat16 dataB[6];
	for (int i = 0; i < 6; i++)
	{
		dataB[i] = lyFloat32ToBfloat16(2.0f);
	}

	lyTensorCreate(&pTensorA, shape, 2, dataA, NULL);
	lyTensorCreate(&pTensorB, shape, 2, dataB, NULL);

	lyTensor* pOutput;
	lyTensorElementwiseMul(&pOutput, pTensorA, pTensorB, pContext);

	float expected[] = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], lyBfloat16ToFloat32(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_TensorMakeTriangularMask(void)
{
	int32_t shape[] = {3, 3};
	lyTensorCreate(&pTensorA, shape, 2, NULL, NULL);
	lyTensorMakeTriangularMask(pTensorA, pContext);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			float expected = j <= i ? 0.0f : -HUGE_VALF;
			float actual   = lyBfloat16ToFloat32(pTensorA->data[i * 3 + j]);
			if (expected == -HUGE_VALF)
			{
				TEST_ASSERT_EQUAL_FLOAT(expected, actual);
			}
			else
			{
				TEST_ASSERT_FLOAT_WITHIN(0.01f, expected, actual);
			}
		}
	}
}

void test_TensorSoftmax(void)
{
	int32_t	   shape[] = {2, 3};
	lyBfloat16 data[]  = {lyFloat32ToBfloat16(1.0f), lyFloat32ToBfloat16(2.0f), lyFloat32ToBfloat16(3.0f), lyFloat32ToBfloat16(4.0f), lyFloat32ToBfloat16(5.0f), lyFloat32ToBfloat16(6.0f)};

	lyTensorCreate(&pTensorA, shape, 2, data, NULL);

	lyTensor* pOutput;
	lyTensorSoftmax(&pOutput, pTensorA, pContext);

	// Check that each row sums to 1
	float sum1 = 0.0f;
	float sum2 = 0.0f;
	for (int i = 0; i < 3; i++)
	{
		sum1 += lyBfloat16ToFloat32(pOutput->data[i]);
		sum2 += lyBfloat16ToFloat32(pOutput->data[i + 3]);
	}

	TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, sum1);
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, sum2);

	// First row expected values (approx): [0.09, 0.24, 0.67]
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.09f, lyBfloat16ToFloat32(pOutput->data[0]));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.24f, lyBfloat16ToFloat32(pOutput->data[1]));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.67f, lyBfloat16ToFloat32(pOutput->data[2]));

	lyTensorDestroy(pOutput);
}

void test_TensorArgmax(void)
{
	int32_t	   shape[] = {1, 5};
	lyBfloat16 data[]  = {lyFloat32ToBfloat16(1.0f), lyFloat32ToBfloat16(3.0f), lyFloat32ToBfloat16(2.0f), lyFloat32ToBfloat16(5.0f), lyFloat32ToBfloat16(4.0f)};

	lyTensorCreate(&pTensorA, shape, 2, data, NULL);

	int32_t maxIdx;
	lyTensorArgmax(&maxIdx, pTensorA, pContext);

	TEST_ASSERT_EQUAL_INT32(3, maxIdx);	 // Index of 5.0f

	// Test with different shape
	int32_t	  shapeVec[] = {5};
	lyTensor* pVec;
	lyTensorCreate(&pVec, shapeVec, 1, data, NULL);

	lyTensorArgmax(&maxIdx, pVec, pContext);
	TEST_ASSERT_EQUAL_INT32(3, maxIdx);

	lyTensorDestroy(pVec);
}

void test_TensorTranspose2D(void)
{
	int32_t	   shape[] = {2, 3};
	lyBfloat16 data[]  = {lyFloat32ToBfloat16(1.0f), lyFloat32ToBfloat16(2.0f), lyFloat32ToBfloat16(3.0f), lyFloat32ToBfloat16(4.0f), lyFloat32ToBfloat16(5.0f), lyFloat32ToBfloat16(6.0f)};

	lyTensorCreate(&pTensorA, shape, 2, data, NULL);

	int32_t	  perm[] = {1, 0};
	lyTensor* pOutput;
	lyTensorTranspose(&pOutput, pTensorA, perm, pContext);

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);

	// Expected matrix:
	// 1 4
	// 2 5
	// 3 6
	float expected[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], lyBfloat16ToFloat32(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_TensorOuter(void)
{
	int32_t shapeA[] = {3};
	int32_t shapeB[] = {2};

	lyBfloat16 dataA[] = {lyFloat32ToBfloat16(1.0f), lyFloat32ToBfloat16(2.0f), lyFloat32ToBfloat16(3.0f)};

	lyBfloat16 dataB[] = {lyFloat32ToBfloat16(4.0f), lyFloat32ToBfloat16(5.0f)};

	lyTensorCreate(&pTensorA, shapeA, 1, dataA, NULL);
	lyTensorCreate(&pTensorB, shapeB, 1, dataB, NULL);

	lyTensor* pOutput;
	lyTensorOuter(&pOutput, pTensorA, pTensorB, pContext);

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);

	// Expected result:
	// [4, 5]
	// [8, 10]
	// [12, 15]
	float expected[] = {4.0f, 5.0f, 8.0f, 10.0f, 12.0f, 15.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], lyBfloat16ToFloat32(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_TensorEmbedding(void)
{
	// Create embedding matrix of shape [4, 3]
	int32_t	   embedShape[] = {4, 3};
	lyBfloat16 embedData[]	= {lyFloat32ToBfloat16(0.1f),
							   lyFloat32ToBfloat16(0.2f),
							   lyFloat32ToBfloat16(0.3f),
							   lyFloat32ToBfloat16(1.1f),
							   lyFloat32ToBfloat16(1.2f),
							   lyFloat32ToBfloat16(1.3f),
							   lyFloat32ToBfloat16(2.1f),
							   lyFloat32ToBfloat16(2.2f),
							   lyFloat32ToBfloat16(2.3f),
							   lyFloat32ToBfloat16(3.1f),
							   lyFloat32ToBfloat16(3.2f),
							   lyFloat32ToBfloat16(3.3f)};

	lyTensorCreate(&pTensorA, embedShape, 2, embedData, NULL);

	// Input tokens: [1, 0, 3, 2]
	int32_t tokens[] = {1, 0, 3, 2};
	int32_t seqLen	 = 4;

	lyTensor* pOutput;
	lyTensorEmbedding(&pOutput, tokens, seqLen, pTensorA, pContext);

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(4, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[1]);

	// Expected output:
	// [1.1, 1.2, 1.3] (token 1)
	// [0.1, 0.2, 0.3] (token 0)
	// [3.1, 3.2, 3.3] (token 3)
	// [2.1, 2.2, 2.3] (token 2)
	float expected[] = {1.1f, 1.2f, 1.3f, 0.1f, 0.2f, 0.3f, 3.1f, 3.2f, 3.3f, 2.1f, 2.2f, 2.3f};

	for (int i = 0; i < 12; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.02f, expected[i], lyBfloat16ToFloat32(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

int main(void)
{
	UNITY_BEGIN();

	RUN_TEST(test_TensorMatMul2D);
	RUN_TEST(test_TensorMatMul3D);
	RUN_TEST(test_TensorScaleAndAdd2D);
	RUN_TEST(test_TensorElementwiseMul);
	RUN_TEST(test_TensorMakeTriangularMask);
	RUN_TEST(test_TensorSoftmax);
	RUN_TEST(test_TensorArgmax);
	RUN_TEST(test_TensorTranspose2D);
	RUN_TEST(test_TensorOuter);
	RUN_TEST(test_TensorEmbedding);

	return UNITY_END();
}