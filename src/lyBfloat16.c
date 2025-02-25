#include "lyBfloat16.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

lyBfloat16 lyFloat32ToBfloat16(float value)
{
	lyBfloat16 result;

	union
	{
		float	 f;
		uint32_t i;
	} u;

	u.f = value;

	result.data = (uint16_t)(u.i >> 16);

	return result;
}

float lyBfloat16ToFloat32(lyBfloat16 value)
{
	union
	{
		float	 f;
		uint32_t i;
	} u;

	u.i = ((uint32_t)value.data) << 16;

	return u.f;
}

lyBfloat16 lyBfloat16Add(lyBfloat16 a, lyBfloat16 b)
{
	float aFloat = lyBfloat16ToFloat32(a);
	float bFloat = lyBfloat16ToFloat32(b);
	return lyFloat32ToBfloat16(aFloat + bFloat);
}

lyBfloat16 lyBfloat16Sub(lyBfloat16 a, lyBfloat16 b)
{
	float aFloat = lyBfloat16ToFloat32(a);
	float bFloat = lyBfloat16ToFloat32(b);
	return lyFloat32ToBfloat16(aFloat - bFloat);
}

lyBfloat16 lyBfloat16Mul(lyBfloat16 a, lyBfloat16 b)
{
	float aFloat = lyBfloat16ToFloat32(a);
	float bFloat = lyBfloat16ToFloat32(b);
	return lyFloat32ToBfloat16(aFloat * bFloat);
}

lyBfloat16 lyBfloat16Div(lyBfloat16 a, lyBfloat16 b)
{
	float aFloat = lyBfloat16ToFloat32(a);
	float bFloat = lyBfloat16ToFloat32(b);
	return lyFloat32ToBfloat16(aFloat / bFloat);
}

bool lyBfloat16Equal(lyBfloat16 a, lyBfloat16 b)
{
	return a.data == b.data;
}

bool lyBfloat16Less(lyBfloat16 a, lyBfloat16 b)
{
	float aFloat = lyBfloat16ToFloat32(a);
	float bFloat = lyBfloat16ToFloat32(b);
	return aFloat < bFloat;
}

bool lyBfloat16Greater(lyBfloat16 a, lyBfloat16 b)
{
	float aFloat = lyBfloat16ToFloat32(a);
	float bFloat = lyBfloat16ToFloat32(b);
	return aFloat > bFloat;
}

lyBfloat16 lyBfloat16Exp(lyBfloat16 value)
{
	float valueFloat = lyBfloat16ToFloat32(value);
	return lyFloat32ToBfloat16(expf(valueFloat));
}

lyBfloat16 lyBfloat16Sqrt(lyBfloat16 value)
{
	float valueFloat = lyBfloat16ToFloat32(value);
	return lyFloat32ToBfloat16(sqrtf(valueFloat));
}

lyBfloat16 lyBfloat16Abs(lyBfloat16 value)
{
	lyBfloat16 result = value;
	result.data &= 0x7FFF;
	return result;
}