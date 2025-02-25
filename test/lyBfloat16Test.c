#include "lyBfloat16.h"

#include "unity.h"

#include <math.h>
#include <stdio.h>

void setUp(void) {}

void tearDown(void) {}

void test_Addition(void)
{
	float a = 1.5f;
	float b = 2.25f;

	lyBfloat16 bfA = lyFloat32ToBfloat16(a);
	lyBfloat16 bfB = lyFloat32ToBfloat16(b);

	lyBfloat16 bfSum = lyBfloat16Add(bfA, bfB);
	float	   sum	 = lyBfloat16ToFloat32(bfSum);

	TEST_ASSERT_FLOAT_WITHIN(0.1f, a + b, sum);
}

void test_Multiplication(void)
{
	float a = 1.5f;
	float b = 2.25f;

	lyBfloat16 bfA = lyFloat32ToBfloat16(a);
	lyBfloat16 bfB = lyFloat32ToBfloat16(b);

	lyBfloat16 bfProduct = lyBfloat16Mul(bfA, bfB);
	float	   product	 = lyBfloat16ToFloat32(bfProduct);

	TEST_ASSERT_FLOAT_WITHIN(0.1f, a * b, product);
}

void test_Division(void)
{
	float a = 1.5f;
	float b = 2.25f;

	lyBfloat16 bfA = lyFloat32ToBfloat16(a);
	lyBfloat16 bfB = lyFloat32ToBfloat16(b);

	lyBfloat16 bfQuotient = lyBfloat16Div(bfA, bfB);
	float	   quotient	  = lyBfloat16ToFloat32(bfQuotient);

	TEST_ASSERT_FLOAT_WITHIN(0.1f, a / b, quotient);
}

void test_Comparison(void)
{
	lyBfloat16 a = lyFloat32ToBfloat16(1.5f);
	lyBfloat16 b = lyFloat32ToBfloat16(2.25f);

	TEST_ASSERT_FALSE(lyBfloat16Equal(a, b));
	TEST_ASSERT_TRUE(lyBfloat16Less(a, b));
	TEST_ASSERT_FALSE(lyBfloat16Greater(a, b));
}

void test_MathFunctions(void)
{
	float	   value   = 1.5f;
	lyBfloat16 bfValue = lyFloat32ToBfloat16(value);

	lyBfloat16 bfExp	 = lyBfloat16Exp(bfValue);
	float	   expResult = lyBfloat16ToFloat32(bfExp);
	TEST_ASSERT_FLOAT_WITHIN(0.1f, expf(value), expResult);

	lyBfloat16 bfSqrt	  = lyBfloat16Sqrt(bfValue);
	float	   sqrtResult = lyBfloat16ToFloat32(bfSqrt);
	TEST_ASSERT_FLOAT_WITHIN(0.1f, sqrtf(value), sqrtResult);

	lyBfloat16 negValue	 = lyFloat32ToBfloat16(-value);
	lyBfloat16 bfAbs	 = lyBfloat16Abs(negValue);
	float	   absResult = lyBfloat16ToFloat32(bfAbs);
	TEST_ASSERT_FLOAT_WITHIN(0.1f, value, absResult);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_Addition);
	RUN_TEST(test_Multiplication);
	RUN_TEST(test_Division);
	RUN_TEST(test_Comparison);
	RUN_TEST(test_MathFunctions);
	return UNITY_END();
}