#include "lyTensor.h"

#include "unity.h"

static lyTensor* pTensor = NULL;

void setUp(void) {}

void tearDown(void)
{
	if (pTensor)
	{
		lyTensorDestroy(pTensor);
		pTensor = NULL;
	}
}

void test_SetTensorShape(void)
{
	int32_t shape[] = {2, 3, 4};
	lyTensorCreate(&pTensor, shape, 3, NULL, NULL);

	TEST_ASSERT_EQUAL_INT32(3, pTensor->rank);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[0]);
	TEST_ASSERT_EQUAL_INT32(3, pTensor->shape[1]);
	TEST_ASSERT_EQUAL_INT32(4, pTensor->shape[2]);
}

void test_SetTensorData(void)
{
	int32_t	   shape[] = {2, 2};
	lyBfloat16 hostData[4];
	for (int i = 0; i < 4; i++)
	{
		hostData[i] = lyFloat32ToBfloat16((float)i);
	}

	lyTensorCreate(&pTensor, shape, 2, hostData, NULL);

	TEST_ASSERT_EQUAL_INT32(2, pTensor->rank);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[1]);

	for (int i = 0; i < 4; i++)
	{
		float expected = (float)i;
		float actual;
		lyTensorGetItemRaw(&actual, pTensor, i);
		TEST_ASSERT_FLOAT_WITHIN(0.01f, expected, actual);
	}
}

void test_ReshapeTensor(void)
{
	int32_t originalShape[] = {2, 3};
	lyTensorCreate(&pTensor, originalShape, 2, NULL, NULL);

	int32_t newShape[] = {3, 2};
	lyTensorReshape(pTensor, newShape, 2);

	TEST_ASSERT_EQUAL_INT32(2, pTensor->rank);
	TEST_ASSERT_EQUAL_INT32(3, pTensor->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pTensor->shape[1]);
}

void test_TensorSlice(void)
{
	int32_t	   shape[] = {4, 2};
	lyBfloat16 hostData[8];
	for (int i = 0; i < 8; i++)
	{
		hostData[i] = lyFloat32ToBfloat16((float)i);
	}

	lyTensorCreate(&pTensor, shape, 2, hostData, NULL);

	lyTensor* pSliced;
	lyTensorSlice(&pSliced, pTensor, 1, 3);

	TEST_ASSERT_EQUAL_INT32(2, pSliced->shape[0]);
	TEST_ASSERT_EQUAL_INT32(2, pSliced->shape[1]);

	lyTensorDestroy(pSliced);
}

void test_TensorGetSetItem(void)
{
	int32_t shape[] = {2, 2};
	lyTensorCreate(&pTensor, shape, 2, NULL, NULL);

	int32_t loc[] = {0, 1};
	lyTensorSetItem(pTensor, loc, 42);

	float value;
	lyTensorGetItem(&value, pTensor, loc);
	TEST_ASSERT_EQUAL_FLOAT(42.0f, value);
}

void test_TensorGetSetFloat(void)
{
	int32_t shape[] = {2};
	lyTensorCreate(&pTensor, shape, 1, NULL, NULL);
	int32_t loc[] = {0};
	lyTensorSetItem(pTensor, loc, 3.14f);

	float value;
	lyTensorGetItem(&value, pTensor, loc);
	TEST_ASSERT_FLOAT_WITHIN(0.03f, 3.14f, value);
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
	return UNITY_END();
}