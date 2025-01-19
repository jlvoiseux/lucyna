// test/lyTensorTest.cu

#include "lyTensor.h"
#include "lyTensorMath.h"
#include "unity.h"

#include <cuda_runtime.h>

static lyTensor* pTensor = NULL;

void setUp(void)
{
	TEST_ASSERT_TRUE(lyCreateTensor(&pTensor));
}

void tearDown(void)
{
	if (pTensor)
	{
		lyDestroyTensor(pTensor);
		pTensor = NULL;
	}
}

void test_SetTensorShape(void)
{
	int32_t shape[] = {2, 3, 4};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensor, shape, 3));

	TEST_ASSERT_EQUAL_INT32(3, pTensor->rank);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[0]);
	TEST_ASSERT_EQUAL_INT32(3, pTensor->shape[1]);
	TEST_ASSERT_EQUAL_INT32(4, pTensor->shape[2]);
}

void test_SetTensorData(void)
{
	int32_t shape[] = {2, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensor, shape, 2));

	nv_bfloat16 hostData[4];
	for (int i = 0; i < 4; i++)
	{
		hostData[i] = __float2bfloat16((float)i);
	}

	TEST_ASSERT_TRUE(lySetTensorData(pTensor, hostData, sizeof(hostData)));

	// Verify data transfer
	nv_bfloat16 verifyData[4];
	cudaMemcpy(verifyData, pTensor->data, sizeof(verifyData), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 4; i++)
	{
		TEST_ASSERT_EQUAL_FLOAT(__bfloat162float(hostData[i]), __bfloat162float(verifyData[i]));
	}
}

void test_ReshapeTensor(void)
{
	int32_t originalShape[] = {2, 3};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensor, originalShape, 2));

	int32_t newShape[] = {3, 2};
	TEST_ASSERT_TRUE(lyReshapeTensor(pTensor, newShape, 2));

	TEST_ASSERT_EQUAL_INT32(2, pTensor->rank);
	TEST_ASSERT_EQUAL_INT32(3, pTensor->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[1]);
}

void test_TensorSlice(void)
{
	int32_t shape[] = {4, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensor, shape, 2));

	nv_bfloat16 hostData[8];
	for (int i = 0; i < 8; i++)
	{
		hostData[i] = __float2bfloat16((float)i);
	}
	TEST_ASSERT_TRUE(lySetTensorData(pTensor, hostData, sizeof(hostData)));

	lyTensor* pSliced;
	TEST_ASSERT_TRUE(lyTensorSlice(&pSliced, pTensor, 1, 3));

	TEST_ASSERT_EQUAL_INT32(2, pSliced->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pSliced->shape[1]);

	lyDestroyTensor(pSliced);
}

void test_TensorGetSetItem(void)
{
	int32_t shape[] = {2, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensor, shape, 2));
	TEST_ASSERT_TRUE(lySetTensorData(pTensor, NULL, 4 * sizeof(nv_bfloat16)));	// Initialize with zeros

	int32_t loc[] = {0, 1};
	TEST_ASSERT_TRUE(lyTensorSetItem(pTensor, loc, 42));

	int32_t value;
	TEST_ASSERT_TRUE(lyTensorGetItem(&value, pTensor, loc));
	TEST_ASSERT_EQUAL_INT32(42, value);
}

void test_TensorGetSetFloat(void)
{
	int32_t shape[] = {2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensor, shape, 1));
	TEST_ASSERT_TRUE(lySetTensorData(pTensor, NULL, 2 * sizeof(nv_bfloat16)));	// Initialize with zeros

	TEST_ASSERT_TRUE(lyTensorSetItemFromFloat32(pTensor, 0, 3.14f));

	float value;
	TEST_ASSERT_TRUE(lyTensorGetItemAsFloat32(&value, pTensor, 0));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 3.14f, value);
}

void test_TensorComplexItem(void)
{
	int32_t shape[] = {2, 2};
	TEST_ASSERT_TRUE(lySetTensorShape(pTensor, shape, 2));
	TEST_ASSERT_TRUE(lySetTensorData(pTensor, NULL, 4 * sizeof(nv_bfloat16)));	// Initialize with zeros

	TEST_ASSERT_TRUE(lyTensorSetComplexItem(pTensor, 0, 1, 1.0f, 2.0f));

	float real, imag;
	TEST_ASSERT_TRUE(lyTensorGetComplexItem(&real, &imag, pTensor, 0, 1));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, real);
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 2.0f, imag);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_SetTensorShape);
	RUN_TEST(test_SetTensorData);
	RUN_TEST(test_ReshapeTensor);
	RUN_TEST(test_TensorSlice);
	RUN_TEST(test_TensorGetSetItem);
	RUN_TEST(test_TensorGetSetFloat);
	RUN_TEST(test_TensorComplexItem);
	return UNITY_END();
}