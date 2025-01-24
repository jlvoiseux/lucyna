// test/lyTensorTest.cu

#include "lyTensor.h"
#include "lyTensorMath.h"
#include "unity.h"

#include <cuda_runtime.h>

static lyTensor* pTensor = NULL;

void setUp(void) {}

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
	lyCreateTensor(&pTensor, shape, 3, NULL, NULL);

	TEST_ASSERT_EQUAL_INT32(3, pTensor->rank);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[0]);
	TEST_ASSERT_EQUAL_INT32(3, pTensor->shape[1]);
	TEST_ASSERT_EQUAL_INT32(4, pTensor->shape[2]);
}

void test_SetTensorData(void)
{
	int32_t		shape[] = {2, 2};
	nv_bfloat16 hostData[4];
	for (int i = 0; i < 4; i++)
	{
		hostData[i] = __float2bfloat16((float)i);
	}

	lyCreateTensor(&pTensor, shape, 2, hostData, NULL);

	TEST_ASSERT_EQUAL_INT32(2, pTensor->rank);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[1]);

	for (int i = 0; i < 4; i++)
	{
		float expected = (float)i;
		float actual;
		TEST_ASSERT_TRUE(lyTensorGetItemAsFloat32(&actual, pTensor, i));
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected, actual);
	}
}

void test_ReshapeTensor(void)
{
	int32_t originalShape[] = {2, 3};
	lyCreateTensor(&pTensor, originalShape, 2, NULL, NULL);

	int32_t newShape[] = {3, 2};
	lyReshapeTensor(pTensor, newShape, 2);

	TEST_ASSERT_EQUAL_INT32(2, pTensor->rank);
	TEST_ASSERT_EQUAL_INT32(3, pTensor->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[1]);
}

void test_TensorSlice(void)
{
	int32_t		shape[] = {4, 2};
	nv_bfloat16 hostData[8];
	for (int i = 0; i < 8; i++)
	{
		hostData[i] = __float2bfloat16((float)i);
	}

	lyCreateTensor(&pTensor, shape, 2, hostData, NULL);

	lyTensor* pSliced;
	lyTensorSlice(&pSliced, pTensor, 1, 3);

	TEST_ASSERT_EQUAL_INT32(2, pSliced->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pSliced->shape[1]);

	lyDestroyTensor(pSliced);
}

void test_TensorGetSetItem(void)
{
	int32_t shape[] = {2, 2};
	lyCreateTensor(&pTensor, shape, 2, NULL, NULL);

	int32_t loc[] = {0, 1};
	TEST_ASSERT_TRUE(lyTensorSetItem(pTensor, loc, 42));

	int32_t value;
	TEST_ASSERT_TRUE(lyTensorGetItem(&value, pTensor, loc));
	TEST_ASSERT_EQUAL_INT32(42, value);
}

void test_TensorGetSetFloat(void)
{
	int32_t shape[] = {2};
	lyCreateTensor(&pTensor, shape, 1, NULL, NULL);
	TEST_ASSERT_TRUE(lyTensorSetItemFromFloat32(pTensor, 0, 3.14f));

	float value;
	TEST_ASSERT_TRUE(lyTensorGetItemAsFloat32(&value, pTensor, 0));
	TEST_ASSERT_FLOAT_WITHIN(0.01f, 3.14f, value);
}

void test_TensorComplexItem(void)
{
	int32_t shape[] = {2, 2};
	lyCreateTensor(&pTensor, shape, 2, NULL, NULL);
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