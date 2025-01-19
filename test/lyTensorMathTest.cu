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

void test_TensorAdd(void)
{
	int32_t shape[] = {2, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 2));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shape, 2));

	nv_bfloat16 dataA[4], dataB[4];
	for (int i = 0; i < 4; i++)
	{
		dataA[i] = __float2bfloat16((float)i);
		dataB[i] = __float2bfloat16((float)(i + 1));
	}

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, dataA, sizeof(dataA)));
	TEST_ASSERT_TRUE(lySetTensorData(pTensorB, dataB, sizeof(dataB)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	TEST_ASSERT_TRUE(lyTensorAdd(&pOutput, pTensorA, pTensorB));
	cudaDeviceSynchronize();

	nv_bfloat16 result[4];
	cudaMemcpy(result, pOutput->data, sizeof(result), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 4; i++)
	{
		float expected = (float)i + (float)(i + 1);
		float actual   = __bfloat162float(result[i]);
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected, actual);
	}

	lyDestroyTensor(pOutput);
}

void test_TensorMatMul(void)
{
	int32_t shapeA[] = {2, 3};
	int32_t shapeB[] = {3, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shapeA, 2));
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorB, shapeB, 2));

	nv_bfloat16 dataA[6], dataB[6];
	for (int i = 0; i < 6; i++)
	{
		dataA[i] = __float2bfloat16((float)i);
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

	lyDestroyTensor(pOutput);
}

void test_TensorScaleAndAdd(void)
{
	int32_t shape[] = {2, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensorA, shape, 2));

	nv_bfloat16 data[4];
	for (int i = 0; i < 4; i++)
	{
		data[i] = __float2bfloat16((float)i);
	}

	TEST_ASSERT_TRUE(lySetTensorData(pTensorA, data, sizeof(data)));
	cudaDeviceSynchronize();

	lyTensor* pOutput;
	float	  scale = 2.0f;
	TEST_ASSERT_TRUE(lyTensorScaleAndAdd(&pOutput, pTensorA, NULL, scale));
	cudaDeviceSynchronize();

	nv_bfloat16 result[4];
	cudaMemcpy(result, pOutput->data, sizeof(result), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 4; i++)
	{
		float expected = (float)i * scale;
		float actual   = __bfloat162float(result[i]);
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected, actual);
	}

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
	cudaDeviceSynchronize();

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

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_TensorAdd);
	RUN_TEST(test_TensorMatMul);
	RUN_TEST(test_TensorScaleAndAdd);
	RUN_TEST(test_TensorElementwiseMul);
	RUN_TEST(test_TensorMakeTriangularMask);
	RUN_TEST(test_TensorArgmax);
	RUN_TEST(test_TensorOuter);
	return UNITY_END();
}