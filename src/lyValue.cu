#include "lyValue.h"

#include <stdio.h>
#include <stdlib.h>

bool lyCreateIntValue(lyValue** ppValue, int64_t val)
{
	if (!ppValue)
	{
		return false;
	}

	lyValue* pValue = (lyValue*)malloc(sizeof(lyValue));
	if (!pValue)
	{
		return false;
	}

	pValue->type	   = LY_VALUE_INT;
	pValue->data.asInt = val;
	*ppValue		   = pValue;

	return true;
}

bool lyCreateBoolValue(lyValue** ppValue, bool val)
{
	if (!ppValue)
	{
		return false;
	}

	lyValue* pValue = (lyValue*)malloc(sizeof(lyValue));
	if (!pValue)
	{
		return false;
	}

	pValue->type		= LY_VALUE_BOOL;
	pValue->data.asBool = val;
	*ppValue			= pValue;

	return true;
}

bool lyCreatePtrValue(lyValue** ppValue, void* ptr)
{
	if (!ppValue)
	{
		return false;
	}

	lyValue* pValue = (lyValue*)malloc(sizeof(lyValue));
	if (!pValue)
	{
		return false;
	}

	pValue->type	   = LY_VALUE_PTR;
	pValue->data.asPtr = ptr;
	*ppValue		   = pValue;

	return true;
}

void lyDestroyValue(lyValue* pValue)
{
	free(pValue);
}

bool lyGetIntValue(const lyValue* pValue, int64_t* pOut)
{
	if (!pValue || pValue->type != LY_VALUE_INT || !pOut)
	{
		return false;
	}

	*pOut = pValue->data.asInt;
	return true;
}

bool lyGetBoolValue(const lyValue* pValue, bool* pOut)
{
	if (!pValue || pValue->type != LY_VALUE_BOOL || !pOut)
	{
		return false;
	}

	*pOut = pValue->data.asBool;
	return true;
}

bool lyGetPtrValue(const lyValue* pValue, void** pOut)
{
	if (!pValue || pValue->type != LY_VALUE_PTR || !pOut)
	{
		return false;
	}

	*pOut = pValue->data.asPtr;
	return true;
}

bool lyCloneValue(lyValue** ppOut, const lyValue* pValue)
{
	if (!ppOut || !pValue)
	{
		return false;
	}

	lyValue* pCopy = (lyValue*)malloc(sizeof(lyValue));
	if (!pCopy)
	{
		return false;
	}

	pCopy->type = pValue->type;

	switch (pValue->type)
	{
		case LY_VALUE_PTR:
			pCopy->data.asPtr = pValue->data.asPtr;
			break;
		case LY_VALUE_INT:
			pCopy->data.asInt = pValue->data.asInt;
			break;
		case LY_VALUE_BOOL:
			pCopy->data.asBool = pValue->data.asBool;
			break;
		default:
			free(pCopy);
			return false;
	}

	*ppOut = pCopy;
	return true;
}

void lyPrintValue(const lyValue* pValue)
{
	if (!pValue)
	{
		printf("NULL");
		return;
	}

	switch (pValue->type)
	{
		case LY_VALUE_NULL:
			printf("NULL");
			break;
		case LY_VALUE_INT:
			printf("INT:%lld", (long long)pValue->data.asInt);
			break;
		case LY_VALUE_BOOL:
			printf("BOOL:%s", pValue->data.asBool ? "true" : "false");
			break;
		case LY_VALUE_PTR:
			printf("PTR:%p", pValue->data.asPtr);
			break;
		default:
			printf("UNKNOWN");
			break;
	}
}