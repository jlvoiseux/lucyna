#pragma once

#include "lyValue.h"

typedef struct lyDict
{
	char**	  keys;
	lyValue** values;
	size_t	  count;
	size_t	  capacity;
} lyDict;

void lyDictCreate(lyDict** ppDict, size_t initCapacity);
void lyDictDestroy(lyDict* pDict);

void lyDictSetValue(lyDict* pDict, const char* key, lyValue* pValue);
void lyDictGetValue(lyValue** ppValue, const lyDict* pDict, const char* key);

void lyDictPrint(const lyDict* pDict);