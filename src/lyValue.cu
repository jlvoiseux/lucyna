#include "lyValue.h"

#include <stdio.h>
#include <stdlib.h>

void lyValueCreateInt(lyValue** ppValue, int64_t val)
{
	lyValue* pValue = (lyValue*)malloc(sizeof(lyValue));

	pValue->type	   = LY_VALUE_INT;
	pValue->data.asInt = val;
	*ppValue		   = pValue;
}

void lyValueCreateBool(lyValue** ppValue, bool val)
{
	lyValue* pValue = (lyValue*)malloc(sizeof(lyValue));

	pValue->type		= LY_VALUE_BOOL;
	pValue->data.asBool = val;
	*ppValue			= pValue;
}

void lyValueCreatePtr(lyValue** ppValue, void* ptr)
{
	lyValue* pValue = (lyValue*)malloc(sizeof(lyValue));

	pValue->type	   = LY_VALUE_PTR;
	pValue->data.asPtr = ptr;
	*ppValue		   = pValue;
}

void lyValueDestroy(lyValue* pValue)
{
	free(pValue);
}

void lyValueGetInt(const lyValue* pValue, int64_t* pOut)
{
	*pOut = pValue->data.asInt;
}

void lyValueGetBool(const lyValue* pValue, bool* pOut)
{
	*pOut = pValue->data.asBool;
}

void lyValueGetPtr(const lyValue* pValue, void** pOut)
{
	*pOut = pValue->data.asPtr;
}

void lyValueClone(lyValue** ppOut, const lyValue* pValue)
{
	lyValue* pCopy = (lyValue*)malloc(sizeof(lyValue));
	pCopy->type	   = pValue->type;

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
	}

	*ppOut = pCopy;
}

void lyValuePrint(const lyValue* pValue)
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