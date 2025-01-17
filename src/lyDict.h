#pragma once

#include "lyValue.h"

typedef struct lyDict
{
	char**	  keys;
	lyValue** values;
	size_t	  count;
	size_t	  capacity;
} lyDict;

bool lyCreateDict(lyDict** ppDict, size_t initCapacity);
void lyDestroyDict(lyDict* pDict);

bool lyDictSetValue(lyDict* pDict, const char* key, lyValue* pValue);
bool lyDictGetValue(lyValue** ppValue, const lyDict* pDict, const char* key);

void lyPrintDict(const lyDict* pDict);