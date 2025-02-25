#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef struct lyBfloat16
{
	uint16_t data;
} lyBfloat16;

lyBfloat16 lyFloat32ToBfloat16(float value);
float	   lyBfloat16ToFloat32(lyBfloat16 value);

lyBfloat16 lyBfloat16Add(lyBfloat16 a, lyBfloat16 b);
lyBfloat16 lyBfloat16Sub(lyBfloat16 a, lyBfloat16 b);
lyBfloat16 lyBfloat16Mul(lyBfloat16 a, lyBfloat16 b);
lyBfloat16 lyBfloat16Div(lyBfloat16 a, lyBfloat16 b);

bool lyBfloat16Equal(lyBfloat16 a, lyBfloat16 b);
bool lyBfloat16Less(lyBfloat16 a, lyBfloat16 b);
bool lyBfloat16Greater(lyBfloat16 a, lyBfloat16 b);

lyBfloat16 lyBfloat16Exp(lyBfloat16 value);
lyBfloat16 lyBfloat16Sqrt(lyBfloat16 value);
lyBfloat16 lyBfloat16Abs(lyBfloat16 value);