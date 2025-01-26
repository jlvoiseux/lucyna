#include "lyTensor.h"
#include "lyTensorMath.h"
#include "unity.h"

static lyTensor* pTensorA = NULL;
static lyTensor* pTensorB = NULL;

void setUp(void) {}

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
}

void test_TensorScaleAndAdd2D(void)
{
	int32_t shape[] = {2, 3};

	nv_bfloat16 dataA[6];
	for (int i = 0; i < 6; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
	}

	nv_bfloat16 dataB[6];
	for (int i = 0; i < 6; i++)
	{
		dataB[i] = __float2bfloat16((float)(i + 1) * 0.5f);
	}

	lyTensorCreate(&pTensorA, shape, 2, dataA, NULL);
	lyTensorCreate(&pTensorB, shape, 2, dataB, NULL);

	lyTensor* pOutput;
	float	  alpha = 2.0f;
	float	  beta	= -1.0f;
	TEST_ASSERT_TRUE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, alpha, beta));

	float expected[] = {1.5f, 3.0f, 4.5f, 6.0f, 7.5f, 9.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], __bfloat162float(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_TensorScaleAndAdd3D(void)
{
	int32_t		 shape[]  = {2, 2, 2};
	size_t		 elements = 8;
	nv_bfloat16* dataA	  = (nv_bfloat16*)malloc(elements * sizeof(nv_bfloat16));
	nv_bfloat16* dataB	  = (nv_bfloat16*)malloc(elements * sizeof(nv_bfloat16));

	for (size_t i = 0; i < elements; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
		dataB[i] = __float2bfloat16((float)(i + 1) * 0.1f);
	}

	lyTensorCreate(&pTensorA, shape, 3, dataA, NULL);
	lyTensorCreate(&pTensorB, shape, 3, dataB, NULL);

	lyTensor* pOutput;
	float	  alpha = 0.5f;
	float	  beta	= 2.0f;
	TEST_ASSERT_TRUE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, alpha, beta));

	for (size_t i = 0; i < elements; i++)
	{
		float expected = 0.5f * (float)(i + 1) + 2.0f * ((float)(i + 1) * 0.1f);
		float actual   = __bfloat162float(pOutput->data[i]);
		TEST_ASSERT_FLOAT_WITHIN(0.05f, expected, actual);
	}

	free(dataA);
	free(dataB);
	lyTensorDestroy(pOutput);
}

void test_TensorScaleAndAddInvalidShapes(void)
{
	int32_t shapeA[] = {2, 3};
	int32_t shapeB[] = {2, 3, 2};
	lyTensorCreate(&pTensorA, shapeA, 2, NULL, NULL);
	lyTensorCreate(&pTensorB, shapeB, 3, NULL, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_FALSE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, 1.0f, 1.0f));

	// Test tensors with same rank but different dimensions
	int32_t shapeC[] = {2, 4};
	lyTensorCreate(&pTensorB, shapeC, 2, NULL, NULL);
	TEST_ASSERT_FALSE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, 1.0f, 1.0f));
}

void test_TensorScaleAndAddRank1Invalid(void)
{
	int32_t shape[] = {3};
	lyTensorCreate(&pTensorA, shape, 1, NULL, NULL);
	lyTensorCreate(&pTensorB, shape, 1, NULL, NULL);
	lyTensor* pOutput;
	TEST_ASSERT_FALSE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, 1.0f, 1.0f));
}

void test_TensorScaleAndAddBroadcast(void)
{
	// Create tensor A with shape (2, 3, 4)
	int32_t shapeA[] = {2, 3, 4};

	// Create tensor B with shape (3, 4)
	int32_t shapeB[] = {3, 4};

	// Initialize tensor A with values 1,2,3,...,24
	nv_bfloat16* dataA = (nv_bfloat16*)malloc(24 * sizeof(nv_bfloat16));
	for (int i = 0; i < 24; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
	}

	// Initialize tensor B with values 0.1,0.2,0.3,...,1.2
	nv_bfloat16* dataB = (nv_bfloat16*)malloc(12 * sizeof(nv_bfloat16));
	for (int i = 0; i < 12; i++)
	{
		dataB[i] = __float2bfloat16((float)(i + 1) * 0.1f);
	}

	lyTensorCreate(&pTensorA, shapeA, 3, dataA, NULL);
	lyTensorCreate(&pTensorB, shapeB, 2, dataB, NULL);

	lyTensor* pOutput;
	float	  alpha = 2.0f;
	float	  beta	= -1.0f;
	TEST_ASSERT_TRUE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, alpha, beta));

	TEST_ASSERT_FLOAT_WITHIN(0.05f, 1.9f, __bfloat162float(pOutput->data[0]));
	TEST_ASSERT_FLOAT_WITHIN(0.05f, 3.8f, __bfloat162float(pOutput->data[1]));
	TEST_ASSERT_FLOAT_WITHIN(0.05f, 5.7f, __bfloat162float(pOutput->data[2]));

	free(dataA);
	free(dataB);
	lyTensorDestroy(pOutput);
}

void test_TensorScaleOnly(void)
{
	int32_t shape[] = {2, 3};

	nv_bfloat16 dataA[6];
	for (int i = 0; i < 6; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
	}

	lyTensorCreate(&pTensorA, shape, 2, dataA, NULL);

	lyTensor* pOutput;
	float	  alpha = 2.0f;
	TEST_ASSERT_TRUE(lyTensorScaleAndAdd(&pOutput, pTensorA, NULL, alpha, 0.0f));  // beta unused when pB is NULL

	// Check that each element was properly scaled
	for (int i = 0; i < 6; i++)
	{
		float expected = (float)(i + 1) * alpha;
		float actual   = __bfloat162float(pOutput->data[i]);
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected, actual);
	}

	lyTensorDestroy(pOutput);
}

void test_MatMul2D(void)
{
	int32_t shapeA[] = {2, 3};
	int32_t shapeB[] = {3, 2};

	nv_bfloat16 dataA[6];
	for (int i = 0; i < 6; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
	}

	nv_bfloat16 dataB[6];
	for (int i = 0; i < 6; i++)
	{
		dataB[i] = __float2bfloat16((float)(i + 1));
	}

	lyTensorCreate(&pTensorA, shapeA, 2, dataA, NULL);
	lyTensorCreate(&pTensorB, shapeB, 2, dataB, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);

	float expected[] = {22.0f, 28.0f, 49.0f, 64.0f};
	for (int i = 0; i < 4; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], __bfloat162float(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_MatMul3D(void)
{
	int32_t shapeA[] = {2, 2, 3};
	int32_t shapeB[] = {2, 3, 2};

	nv_bfloat16 dataA[12], dataB[12];
	for (int i = 0; i < 12; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
		dataB[i] = __float2bfloat16((float)(i + 1));
	}

	lyTensorCreate(&pTensorA, shapeA, 3, dataA, NULL);
	lyTensorCreate(&pTensorB, shapeB, 3, dataB, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));

	TEST_ASSERT_EQUAL_INT32(3, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[2]);

	lyTensorDestroy(pOutput);
}

void test_MatMulInvalidShapes(void)
{
	int32_t shapeA[] = {2, 3};
	int32_t shapeB[] = {2, 2};
	lyTensorCreate(&pTensorA, shapeA, 2, NULL, NULL);
	lyTensorCreate(&pTensorB, shapeB, 2, NULL, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_FALSE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));
}

void test_MatMulDifferentRanks(void)
{
	int32_t shapeA[] = {2, 2, 3};
	int32_t shapeB[] = {3, 2};	// Different rank
	lyTensorCreate(&pTensorA, shapeA, 3, NULL, NULL);
	lyTensorCreate(&pTensorB, shapeB, 2, NULL, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_FALSE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));
}

void test_MatMul4D(void)
{
	int32_t shapeA[] = {2, 2, 2, 3};
	int32_t shapeB[] = {2, 2, 3, 2};

	int32_t		 sizeA = 2 * 2 * 2 * 3;
	int32_t		 sizeB = 2 * 2 * 3 * 2;
	nv_bfloat16* dataA = (nv_bfloat16*)malloc(sizeA * sizeof(nv_bfloat16));
	nv_bfloat16* dataB = (nv_bfloat16*)malloc(sizeB * sizeof(nv_bfloat16));

	for (int i = 0; i < sizeA; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
	}
	for (int i = 0; i < sizeB; i++)
	{
		dataB[i] = __float2bfloat16((float)(i + 1));
	}

	lyTensorCreate(&pTensorA, shapeA, 4, dataA, NULL);
	lyTensorCreate(&pTensorB, shapeB, 4, dataB, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));

	TEST_ASSERT_EQUAL_INT32(4, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[2]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[3]);

	free(dataA);
	free(dataB);
	lyTensorDestroy(pOutput);
}

void test_TensorElementwiseMul(void)
{
	int32_t shape[] = {2, 2};

	nv_bfloat16 dataA[4], dataB[4];
	for (int i = 0; i < 4; i++)
	{
		dataA[i] = __float2bfloat16((float)i);
		dataB[i] = __float2bfloat16(2.0f);
	}

	lyTensorCreate(&pTensorA, shape, 2, dataA, NULL);
	lyTensorCreate(&pTensorB, shape, 2, dataB, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorElementwiseMul(&pOutput, pTensorA, pTensorB));

	for (int i = 0; i < 4; i++)
	{
		float expected = (float)i * 2.0f;
		float actual   = __bfloat162float(pOutput->data[i]);
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected, actual);
	}

	lyTensorDestroy(pOutput);
}

void test_TensorMakeTriangularMask(void)
{
	int32_t shape[] = {3, 3};

	lyTensorCreate(&pTensorA, shape, 2, NULL, NULL);

	TEST_ASSERT_TRUE(lyTensorMakeTriangularMask(pTensorA));

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			float expected = j <= i ? 0.0f : -INFINITY;
			float actual   = __bfloat162float(pTensorA->data[i * 3 + j]);
			if (expected == -INFINITY)
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

void test_TensorArgmax(void)
{
	int32_t		shape[] = {2, 3};
	nv_bfloat16 data[]	= {__float2bfloat16(1.0f), __float2bfloat16(3.0f), __float2bfloat16(2.0f), __float2bfloat16(0.0f), __float2bfloat16(5.0f), __float2bfloat16(4.0f)};
	lyTensorCreate(&pTensorA, shape, 2, data, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorArgmax(&pOutput, pTensorA, 1));

	TEST_ASSERT_EQUAL_INT32(1, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);

	TEST_ASSERT_EQUAL_INT32(1, (int32_t)__bfloat162float(pOutput->data[0]));  // max at index 1 in first row
	TEST_ASSERT_EQUAL_INT32(1, (int32_t)__bfloat162float(pOutput->data[1]));  // max at index 1 in second row

	lyTensorDestroy(pOutput);
}

void test_TensorSoftmax(void)
{
	int32_t		shape[] = {2, 3};
	nv_bfloat16 data[]	= {__float2bfloat16(1.0f), __float2bfloat16(2.0f), __float2bfloat16(3.0f), __float2bfloat16(4.0f), __float2bfloat16(5.0f), __float2bfloat16(6.0f)};
	lyTensorCreate(&pTensorA, shape, 2, data, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorSoftmax(&pOutput, pTensorA, 1));

	float sum1 = 0.0f;
	for (int i = 0; i < 3; i++)
	{
		sum1 += __bfloat162float(pOutput->data[i]);
	}
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, sum1);

	float sum2 = 0.0f;
	for (int i = 3; i < 6; i++)
	{
		sum2 += __bfloat162float(pOutput->data[i]);
	}
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, sum2);

	float expected1[] = {0.0900f, 0.2447f, 0.6652f};
	for (int i = 0; i < 3; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected1[i], __bfloat162float(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);

	int32_t		shape3d[] = {2, 2, 2};
	nv_bfloat16 data3d[]  = {__float2bfloat16(1.0f), __float2bfloat16(2.0f), __float2bfloat16(3.0f), __float2bfloat16(4.0f), __float2bfloat16(5.0f), __float2bfloat16(6.0f), __float2bfloat16(7.0f), __float2bfloat16(8.0f)};
	lyTensorCreate(&pTensorA, shape3d, 3, data3d, NULL);

	TEST_ASSERT_TRUE(lyTensorSoftmax(&pOutput, pTensorA, 1));

	TEST_ASSERT_EQUAL_INT32(3, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[2]);

	for (int i = 0; i < 2; i++)
	{
		for (int k = 0; k < 2; k++)
		{
			float sum = 0.0f;
			for (int j = 0; j < 2; j++)
			{
				sum += __bfloat162float(pOutput->data[i * 4 + j * 2 + k]);
			}
			TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, sum);
		}
	}

	lyTensorDestroy(pOutput);
}

void test_TensorOuter(void)
{
	int32_t shapeA[] = {2};
	int32_t shapeB[] = {3};

	nv_bfloat16 dataA[] = {__float2bfloat16(1.0f), __float2bfloat16(2.0f)};
	nv_bfloat16 dataB[] = {__float2bfloat16(3.0f), __float2bfloat16(4.0f), __float2bfloat16(5.0f)};

	lyTensorCreate(&pTensorA, shapeA, 1, dataA, NULL);
	lyTensorCreate(&pTensorB, shapeB, 1, dataB, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorOuter(&pOutput, pTensorA, pTensorB));

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[1]);

	float expected[6] = {3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 10.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], __bfloat162float(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_TensorEmbedding(void)
{
	// Create input tokens tensor [3]
	int32_t tokenShape[] = {3};

	// Create embedding matrix [4, 2]
	int32_t embeddingShape[] = {4, 2};

	// Set up token IDs: [1, 0, 2]
	nv_bfloat16 tokenData[3];
	tokenData[0] = __float2bfloat16(1.0f);
	tokenData[1] = __float2bfloat16(0.0f);
	tokenData[2] = __float2bfloat16(2.0f);

	// Set up embedding matrix:
	// [[0.0, 0.1],
	//  [1.0, 1.1],
	//  [2.0, 2.1],
	//  [3.0, 3.1]]
	nv_bfloat16 embeddingData[8];
	for (int i = 0; i < 4; i++)
	{
		embeddingData[i * 2]	 = __float2bfloat16((float)i);
		embeddingData[i * 2 + 1] = __float2bfloat16((float)i + 0.1f);
	}

	lyTensorCreate(&pTensorA, tokenShape, 1, tokenData, NULL);
	lyTensorCreate(&pTensorB, embeddingShape, 2, embeddingData, NULL);

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorEmbedding(&pOutput, pTensorA, pTensorB));

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[0]);	// sequence length
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);	// embedding dimension

	// Expected sequence:
	// Token 1 -> [1.0, 1.1]
	// Token 0 -> [0.0, 0.1]
	// Token 2 -> [2.0, 2.1]
	float expected[6] = {1.0f, 1.1f, 0.0f, 0.1f, 2.0f, 2.1f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], __bfloat162float(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_TensorTranspose(void)
{
	int32_t		 shape[] = {2, 3};
	nv_bfloat16* data	 = (nv_bfloat16*)malloc(6 * sizeof(nv_bfloat16));
	for (int i = 0; i < 6; i++)
	{
		data[i] = __float2bfloat16((float)i);
	}
	lyTensorCreate(&pTensorA, shape, 2, data, NULL);
	free(data);

	lyTensor* pOutput;
	int32_t	  perm[] = {1, 0};

	TEST_ASSERT_TRUE(lyTensorTranspose(&pOutput, pTensorA, perm));

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);

	// Expected after transpose:
	// 0 3
	// 1 4
	// 2 5
	float expected[] = {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], __bfloat162float(pOutput->data[i]));
	}

	lyTensorDestroy(pOutput);
}

void test_TensorTranspose3D(void)
{
	int32_t		 shape[] = {2, 3, 4};
	nv_bfloat16* data	 = (nv_bfloat16*)malloc(24 * sizeof(nv_bfloat16));
	for (int i = 0; i < 24; i++)
	{
		data[i] = __float2bfloat16((float)i);
	}
	lyTensorCreate(&pTensorA, shape, 3, data, NULL);
	free(data);

	lyTensor* pOutput;
	int32_t	  perm[] = {0, 2, 1};
	TEST_ASSERT_TRUE(lyTensorTranspose(&pOutput, pTensorA, perm));

	TEST_ASSERT_EQUAL_INT32(3, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(4, pOutput->shape[1]);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[2]);

	TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, __bfloat162float(pOutput->data[0]));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 4.0f, __bfloat162float(pOutput->data[1]));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 8.0f, __bfloat162float(pOutput->data[2]));

	lyTensorDestroy(pOutput);
}

void test_TensorTranspose3DLarge(void)
{
	int32_t shape[]	 = {32, 32, 128};
	size_t	elements = 32 * 32 * 128;

	nv_bfloat16* data = (nv_bfloat16*)malloc(elements * sizeof(nv_bfloat16));
	TEST_ASSERT_NOT_NULL(data);

	for (size_t i = 0; i < elements; i++)
	{
		data[i] = __float2bfloat16((float)(i + 1));
	}

	lyTensorCreate(&pTensorA, shape, 3, data, NULL);

	int32_t	  perm[] = {1, 0, 2};
	lyTensor* pTransposed;
	TEST_ASSERT_TRUE(lyTensorTranspose(&pTransposed, pTensorA, perm));

	TEST_ASSERT_EQUAL_INT32(3, pTransposed->rank);
	TEST_ASSERT_EQUAL_INT32(32, pTransposed->shape[0]);
	TEST_ASSERT_EQUAL_INT32(32, pTransposed->shape[1]);
	TEST_ASSERT_EQUAL_INT32(128, pTransposed->shape[2]);

	for (int i = 0; i < 32; i++)
	{
		for (int j = 0; j < 32; j++)
		{
			for (int k = 0; k < 128; k++)
			{
				size_t originalIndex   = i * 32 * 128 + j * 128 + k;
				size_t transposedIndex = j * 32 * 128 + i * 128 + k;
				TEST_ASSERT_FLOAT_WITHIN(0.01f, __bfloat162float(data[originalIndex]), __bfloat162float(pTransposed->data[transposedIndex]));
			}
		}
	}

	free(data);
	lyTensorDestroy(pTransposed);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_TensorScaleAndAdd2D);
	RUN_TEST(test_TensorScaleAndAdd3D);
	RUN_TEST(test_TensorScaleAndAddInvalidShapes);
	RUN_TEST(test_TensorScaleAndAddRank1Invalid);
	RUN_TEST(test_TensorScaleAndAddBroadcast);
	RUN_TEST(test_TensorScaleOnly);
	RUN_TEST(test_MatMul2D);
	RUN_TEST(test_MatMul3D);
	RUN_TEST(test_MatMulInvalidShapes);
	RUN_TEST(test_MatMulDifferentRanks);
	RUN_TEST(test_MatMul4D);
	RUN_TEST(test_TensorElementwiseMul);
	RUN_TEST(test_TensorMakeTriangularMask);
	RUN_TEST(test_TensorArgmax);
	RUN_TEST(test_TensorSoftmax);
	RUN_TEST(test_TensorOuter);
	RUN_TEST(test_TensorEmbedding);
	RUN_TEST(test_TensorTranspose);
	RUN_TEST(test_TensorTranspose3D);
	RUN_TEST(test_TensorTranspose3DLarge);
	return UNITY_END();
}