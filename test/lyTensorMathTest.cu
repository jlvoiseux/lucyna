#include "lyTensor.h"
#include "lyTensorMath.h"
#include "unity.h"

static lyTensor* pTensorA = NULL;
static lyTensor* pTensorB = NULL;

void setUp(void)
{
	TEST_ASSERT_TRUE(lyCreateTensor(&pTensorA));
	TEST_ASSERT_TRUE(lyCreateTensor(&pTensorB));
}

void tearDown(void)
{
	if (pTensorA)
	{
		lyDestroyTensor(pTensorA);
		pTensorA = NULL;
	}
	if (pTensorB)
	{
		lyDestroyTensor(pTensorB);
		pTensorB = NULL;
	}
}

void test_TensorScaleAndAdd2D(void)
{
	// Test with 2D tensors
	int32_t shape[] = {2, 3};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 2));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shape, 2));

	// Initialize tensor A with values 1,2,3,4,5,6
	nv_bfloat16 dataA[6];
	for (int i = 0; i < 6; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
	}

	// Initialize tensor B with values 0.5,1.0,1.5,2.0,2.5,3.0
	nv_bfloat16 dataB[6];
	for (int i = 0; i < 6; i++)
	{
		dataB[i] = __float2bfloat16((float)(i + 1) * 0.5f);
	}

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, dataA, sizeof(dataA)));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, dataB, sizeof(dataB)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	float	  alpha = 2.0f;
	float	  beta	= -1.0f;
	TEST_ASSERT_TRUE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, alpha, beta));
	cudaDeviceSynchronize();

	nv_bfloat16 result[6];
	cudaMemcpy(result, pOutput->data, sizeof(result), cudaMemcpyDeviceToHost);

	// Expected: 2*A - B
	float expected[] = {1.5f, 3.0f, 4.5f, 6.0f, 7.5f, 9.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], __bfloat162float(result[i]));
	}

	lyDestroyTensor(pOutput);
}

void test_TensorScaleAndAdd3D(void)
{
	// Test with 3D tensors
	int32_t shape[] = {2, 2, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 3));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shape, 3));

	size_t		 elements = 8;
	nv_bfloat16* dataA	  = (nv_bfloat16*)malloc(elements * sizeof(nv_bfloat16));
	nv_bfloat16* dataB	  = (nv_bfloat16*)malloc(elements * sizeof(nv_bfloat16));

	for (size_t i = 0; i < elements; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
		dataB[i] = __float2bfloat16((float)(i + 1) * 0.1f);
	}

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, dataA, elements * sizeof(nv_bfloat16)));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, dataB, elements * sizeof(nv_bfloat16)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	float	  alpha = 0.5f;
	float	  beta	= 2.0f;
	TEST_ASSERT_TRUE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, alpha, beta));
	cudaDeviceSynchronize();

	nv_bfloat16* result = (nv_bfloat16*)malloc(elements * sizeof(nv_bfloat16));
	cudaMemcpy(result, pOutput->data, elements * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < elements; i++)
	{
		float expected = 0.5f * (float)(i + 1) + 2.0f * ((float)(i + 1) * 0.1f);
		float actual   = __bfloat162float(result[i]);
		TEST_ASSERT_FLOAT_WITHIN(0.05f, expected, actual);
	}

	free(dataA);
	free(dataB);
	free(result);
	lyDestroyTensor(pOutput);
}

void test_TensorScaleAndAddInvalidShapes(void)
{
	int32_t shapeA[] = {2, 3};
	int32_t shapeB[] = {2, 3, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shapeA, 2));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeB, 3));

	lyTensor* pOutput;
	TEST_ASSERT_FALSE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, 1.0f, 1.0f));

	// Test tensors with same rank but different dimensions
	int32_t shapeC[] = {2, 4};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeC, 2));
	TEST_ASSERT_FALSE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, 1.0f, 1.0f));
}

void test_TensorScaleAndAddRank1Invalid(void)
{
	int32_t shape[] = {3};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 1));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shape, 1));

	lyTensor* pOutput;
	TEST_ASSERT_FALSE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, 1.0f, 1.0f));
}

void test_TensorScaleAndAddBroadcast(void)
{
	// Create tensor A with shape (2, 3, 4)
	int32_t shapeA[] = {2, 3, 4};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shapeA, 3));

	// Create tensor B with shape (3, 4)
	int32_t shapeB[] = {3, 4};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeB, 2));

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

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, dataA, 24 * sizeof(nv_bfloat16)));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, dataB, 12 * sizeof(nv_bfloat16)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	float	  alpha = 2.0f;
	float	  beta	= -1.0f;
	TEST_ASSERT_TRUE(lyTensorScaleAndAdd(&pOutput, pTensorA, pTensorB, alpha, beta));
	cudaDeviceSynchronize();

	nv_bfloat16* result = (nv_bfloat16*)malloc(24 * sizeof(nv_bfloat16));
	cudaMemcpy(result, pOutput->data, 24 * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

	// Verify first few elements
	TEST_ASSERT_FLOAT_WITHIN(0.05f, 1.9f, __bfloat162float(result[0]));
	TEST_ASSERT_FLOAT_WITHIN(0.05f, 3.8f, __bfloat162float(result[1]));
	TEST_ASSERT_FLOAT_WITHIN(0.05f, 5.7f, __bfloat162float(result[2]));

	free(dataA);
	free(dataB);
	free(result);
	lyDestroyTensor(pOutput);
}

void test_MatMul2D(void)
{
	int32_t shapeA[] = {2, 3};
	int32_t shapeB[] = {3, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shapeA, 2));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeB, 2));

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

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, dataA, sizeof(dataA)));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, dataB, sizeof(dataB)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));
	cudaDeviceSynchronize();

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);

	nv_bfloat16 result[4];
	cudaMemcpy(result, pOutput->data, sizeof(result), cudaMemcpyDeviceToHost);

	float expected[] = {22.0f, 28.0f, 49.0f, 64.0f};
	for (int i = 0; i < 4; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.1f, expected[i], __bfloat162float(result[i]));
	}

	lyDestroyTensor(pOutput);
}

void test_MatMul3D(void)
{
	int32_t shapeA[] = {2, 2, 3};
	int32_t shapeB[] = {2, 3, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shapeA, 3));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeB, 3));

	nv_bfloat16 dataA[12], dataB[12];
	for (int i = 0; i < 12; i++)
	{
		dataA[i] = __float2bfloat16((float)(i + 1));
		dataB[i] = __float2bfloat16((float)(i + 1));
	}

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, dataA, sizeof(dataA)));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, dataB, sizeof(dataB)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));
	cudaDeviceSynchronize();

	TEST_ASSERT_EQUAL_INT32(3, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[2]);

	lyDestroyTensor(pOutput);
}

void test_MatMulInvalidShapes(void)
{
	int32_t shapeA[] = {2, 3};
	int32_t shapeB[] = {2, 2};	// Invalid: inner dimensions don't match
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shapeA, 2));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeB, 2));

	lyTensor* pOutput;
	TEST_ASSERT_FALSE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));
}

void test_MatMulDifferentRanks(void)
{
	int32_t shapeA[] = {2, 2, 3};
	int32_t shapeB[] = {3, 2};	// Different rank
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shapeA, 3));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeB, 2));

	lyTensor* pOutput;
	TEST_ASSERT_FALSE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));
}

void test_MatMul4D(void)
{
	int32_t shapeA[] = {2, 2, 2, 3};
	int32_t shapeB[] = {2, 2, 3, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shapeA, 4));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeB, 4));

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

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, dataA, sizeA * sizeof(nv_bfloat16)));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, dataB, sizeB * sizeof(nv_bfloat16)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorMatMul(&pOutput, pTensorA, pTensorB));
	cudaDeviceSynchronize();

	TEST_ASSERT_EQUAL_INT32(4, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[2]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[3]);

	free(dataA);
	free(dataB);
	lyDestroyTensor(pOutput);
}

void test_TensorElementwiseMul(void)
{
	int32_t shape[] = {2, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 2));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shape, 2));

	nv_bfloat16 dataA[4], dataB[4];
	for (int i = 0; i < 4; i++)
	{
		dataA[i] = __float2bfloat16((float)i);
		dataB[i] = __float2bfloat16(2.0f);
	}

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, dataA, sizeof(dataA)));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, dataB, sizeof(dataB)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorElementwiseMul(&pOutput, pTensorA, pTensorB));
	cudaDeviceSynchronize();

	nv_bfloat16 result[4];
	cudaMemcpy(result, pOutput->data, sizeof(result), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 4; i++)
	{
		float expected = (float)i * 2.0f;
		float actual   = __bfloat162float(result[i]);
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected, actual);
	}

	lyDestroyTensor(pOutput);
}

void test_TensorMakeTriangularMask(void)
{
	int32_t shape[] = {3, 3};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 2));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, NULL, 9 * sizeof(nv_bfloat16)));

	TEST_ASSERT_TRUE(lyTensorMakeTriangularMask(pTensorA));
	cudaDeviceSynchronize();

	nv_bfloat16 result[9];
	cudaMemcpy(result, pTensorA->data, sizeof(result), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			float expected = j <= i ? 0.0f : -INFINITY;
			float actual   = __bfloat162float(result[i * 3 + j]);
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
	int32_t shape[] = {2, 3};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 2));

	nv_bfloat16 data[] = {__float2bfloat16(1.0f), __float2bfloat16(3.0f), __float2bfloat16(2.0f), __float2bfloat16(0.0f), __float2bfloat16(5.0f), __float2bfloat16(4.0f)};
	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, data, sizeof(data)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorArgmax(&pOutput, pTensorA, 1));
	cudaDeviceSynchronize();

	TEST_ASSERT_EQUAL_INT32(1, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);

	nv_bfloat16 result[2];
	cudaMemcpy(result, pOutput->data, sizeof(result), cudaMemcpyDeviceToHost);

	TEST_ASSERT_EQUAL_INT32(1, (int32_t)__bfloat162float(result[0]));  // max at index 1 in first row
	TEST_ASSERT_EQUAL_INT32(1, (int32_t)__bfloat162float(result[1]));  // max at index 1 in second row

	lyDestroyTensor(pOutput);
}

void test_TensorOuter(void)
{
	int32_t shapeA[] = {2};
	int32_t shapeB[] = {3};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shapeA, 1));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeB, 1));

	nv_bfloat16 dataA[] = {__float2bfloat16(1.0f), __float2bfloat16(2.0f)};
	nv_bfloat16 dataB[] = {__float2bfloat16(3.0f), __float2bfloat16(4.0f), __float2bfloat16(5.0f)};

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, dataA, sizeof(dataA)));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, dataB, sizeof(dataB)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorOuter(&pOutput, pTensorA, pTensorB));
	cudaDeviceSynchronize();

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[1]);

	nv_bfloat16 result[6];
	cudaMemcpy(result, pOutput->data, sizeof(result), cudaMemcpyDeviceToHost);

	float expected[6] = {3.0f, 4.0f, 5.0f, 6.0f, 8.0f, 10.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], __bfloat162float(result[i]));
	}

	lyDestroyTensor(pOutput);
}

void test_TensorEmbedding(void)
{
	// Create input tokens tensor [3]
	int32_t tokenShape[] = {3};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, tokenShape, 1));

	// Create embedding matrix [4, 2]
	int32_t embeddingShape[] = {4, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, embeddingShape, 2));

	// Set up token IDs: [1, 0, 2]
	nv_bfloat16 tokenData[3];
	tokenData[0] = __float2bfloat16(1.0f);
	tokenData[1] = __float2bfloat16(0.0f);
	tokenData[2] = __float2bfloat16(2.0f);
	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, tokenData, sizeof(tokenData)));

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
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, embeddingData, sizeof(embeddingData)));
	cudaDeviceSynchronize();

	// Perform embedding lookup
	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorEmbedding(&pOutput, pTensorA, pTensorB));
	cudaDeviceSynchronize();

	// Check output shape
	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[0]);	// sequence length
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);	// embedding dimension

	// Check output values
	nv_bfloat16 result[6];
	cudaMemcpy(result, pOutput->data, sizeof(result), cudaMemcpyDeviceToHost);

	// Expected sequence:
	// Token 1 -> [1.0, 1.1]
	// Token 0 -> [0.0, 0.1]
	// Token 2 -> [2.0, 2.1]
	float expected[6] = {1.0f, 1.1f, 0.0f, 0.1f, 2.0f, 2.1f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], __bfloat162float(result[i]));
	}

	lyDestroyTensor(pOutput);
}

void test_TensorTranspose(void)
{
	int32_t shape[] = {2, 3};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 2));

	nv_bfloat16* data = (nv_bfloat16*)malloc(6 * sizeof(nv_bfloat16));
	for (int i = 0; i < 6; i++)
	{
		data[i] = __float2bfloat16((float)i);
	}
	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, data, 6 * sizeof(nv_bfloat16)));
	free(data);

	lyTensor* pOutput;
	int32_t	  perm[] = {1, 0};

	TEST_ASSERT_TRUE(lyTensorTranspose(&pOutput, pTensorA, perm));

	TEST_ASSERT_EQUAL_INT32(2, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[1]);

	nv_bfloat16 result[6];
	cudaMemcpy(result, pOutput->data, 6 * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

	// Expected after transpose:
	// 0 3
	// 1 4
	// 2 5
	float expected[] = {0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f};
	for (int i = 0; i < 6; i++)
	{
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected[i], __bfloat162float(result[i]));
	}

	lyDestroyTensor(pOutput);
}

void test_TensorTranspose3D(void)
{
	int32_t shape[] = {2, 3, 4};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 3));

	nv_bfloat16* data = (nv_bfloat16*)malloc(24 * sizeof(nv_bfloat16));
	for (int i = 0; i < 24; i++)
	{
		data[i] = __float2bfloat16((float)i);
	}
	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, data, 24 * sizeof(nv_bfloat16)));
	free(data);

	lyTensor* pOutput;
	int32_t	  perm[] = {0, 2, 1};
	TEST_ASSERT_TRUE(lyTensorTranspose(&pOutput, pTensorA, perm));

	TEST_ASSERT_EQUAL_INT32(3, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(2, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(4, pOutput->shape[1]);
	TEST_ASSERT_EQUAL_INT32(3, pOutput->shape[2]);

	nv_bfloat16* result = (nv_bfloat16*)malloc(24 * sizeof(nv_bfloat16));
	cudaMemcpy(result, pOutput->data, 24 * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

	TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, __bfloat162float(result[0]));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 4.0f, __bfloat162float(result[1]));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 8.0f, __bfloat162float(result[2]));

	free(result);
	lyDestroyTensor(pOutput);
}

void test_TensorTransposeLarge(void)
{
	const int32_t seqLen   = 32;
	const int32_t nKVHeads = 32;
	const int32_t headDim  = 128;

	int32_t shape[] = {seqLen, nKVHeads, headDim};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 3));

	size_t		 totalElements = seqLen * nKVHeads * headDim;
	nv_bfloat16* data		   = (nv_bfloat16*)malloc(totalElements * sizeof(nv_bfloat16));

	// Initialize with more predictable values for validation
	for (int i = 0; i < seqLen; i++)
	{
		for (int j = 0; j < nKVHeads; j++)
		{
			for (int k = 0; k < headDim; k++)
			{
				size_t idx = i * (nKVHeads * headDim) + j * headDim + k;
				data[idx]  = __float2bfloat16((float)(j + k));	// Value depends on last two dims
			}
		}
	}

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, data, totalElements * sizeof(nv_bfloat16)));
	free(data);

	lyTensor* pOutput;
	int32_t	  perm[] = {0, 2, 1};
	TEST_ASSERT_TRUE(lyTensorTranspose(&pOutput, pTensorA, perm));
	cudaDeviceSynchronize();

	TEST_ASSERT_EQUAL_INT32(3, pOutput->rank);
	TEST_ASSERT_EQUAL_INT32(seqLen, pOutput->shape[0]);
	TEST_ASSERT_EQUAL_INT32(headDim, pOutput->shape[1]);
	TEST_ASSERT_EQUAL_INT32(nKVHeads, pOutput->shape[2]);

	nv_bfloat16* result = (nv_bfloat16*)malloc(totalElements * sizeof(nv_bfloat16));
	cudaMemcpy(result, pOutput->data, totalElements * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);

	// Validate specific positions where we know the expected values
	int32_t testPositions[][3] = {
		{0, 0, 0},	// Should be same value
		{0, 1, 0},	// Should match value at (0,0,1) in original
		{0, 0, 1},	// Should match value at (0,1,0) in original
		{1, 2, 3},	// Should match value at (1,3,2) in original
	};

	for (int i = 0; i < 4; i++)
	{
		int32_t x = testPositions[i][0];
		int32_t y = testPositions[i][1];
		int32_t z = testPositions[i][2];

		size_t originalIdx	 = x * (nKVHeads * headDim) + z * headDim + y;
		size_t transposedIdx = x * (headDim * nKVHeads) + y * nKVHeads + z;

		float originalVal	= __bfloat162float(data[originalIdx]);
		float transposedVal = __bfloat162float(result[transposedIdx]);
		TEST_ASSERT_FLOAT_WITHIN(0.01f, originalVal, transposedVal);
	}

	free(result);
	lyDestroyTensor(pOutput);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_TensorScaleAndAdd2D);
	RUN_TEST(test_TensorScaleAndAdd3D);
	RUN_TEST(test_TensorScaleAndAddInvalidShapes);
	RUN_TEST(test_TensorScaleAndAddRank1Invalid);
	RUN_TEST(test_TensorScaleAndAddBroadcast);
	RUN_TEST(test_MatMul2D);
	RUN_TEST(test_MatMul3D);
	RUN_TEST(test_MatMulInvalidShapes);
	RUN_TEST(test_MatMulDifferentRanks);
	RUN_TEST(test_MatMul4D);
	RUN_TEST(test_TensorElementwiseMul);
	RUN_TEST(test_TensorMakeTriangularMask);
	RUN_TEST(test_TensorArgmax);
	RUN_TEST(test_TensorOuter);
	RUN_TEST(test_TensorEmbedding);
	RUN_TEST(test_TensorTranspose);
	RUN_TEST(test_TensorTranspose3D);
	RUN_TEST(test_TensorTransposeLarge);
	return UNITY_END();
}