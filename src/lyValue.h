#pragma once

#include <stdbool.h>
#include <stdint.h>

typedef enum lyValueType
{
	LY_VALUE_NULL,
	LY_VALUE_INT,
	LY_VALUE_BOOL,
	LY_VALUE_PTR
} lyValueType;

typedef struct lyValue
{
	lyValueType type;
	union
	{
		int64_t asInt;
		bool	asBool;
		void*	asPtr;
	} data;
} lyValue;

void lyValueCreateInt(lyValue** ppValue, int64_t val);
void lyValueCreateBool(lyValue** ppValue, bool val);
void lyValueCreatePtr(lyValue** ppValue, void* ptr);
void lyValueDestroy(lyValue* pValue);

void lyValueGetInt(const lyValue* pValue, int64_t* pOut);
void lyValueGetBool(const lyValue* pValue, bool* pOut);
void lyValueGetPtr(const lyValue* pValue, void** pOut);

void lyValueClone(lyValue** ppOut, const lyValue* pValue);

void lyValuePrint(const lyValue* pValue);