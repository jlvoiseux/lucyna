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

bool lyCreateIntValue(lyValue** ppValue, int64_t val);
bool lyCreateBoolValue(lyValue** ppValue, bool val);
bool lyCreatePtrValue(lyValue** ppValue, void* ptr);
void lyDestroyValue(lyValue* pValue);

bool lyGetIntValue(const lyValue* pValue, int64_t* pOut);
bool lyGetBoolValue(const lyValue* pValue, bool* pOut);
bool lyGetPtrValue(const lyValue* pValue, void** pOut);

bool lyCloneValue(lyValue** ppOut, const lyValue* pValue);

void lyPrintValue(const lyValue* pValue);