#include "lyDict.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool lyCreateDict(lyDict** ppDict, size_t initCapacity)
{
	if (!ppDict)
	{
		return false;
	}

	lyDict* pDict = (lyDict*)malloc(sizeof(lyDict));
	if (!pDict)
	{
		return false;
	}

	char**	  keys	 = (char**)malloc(sizeof(char*) * initCapacity);
	lyValue** values = (lyValue**)malloc(sizeof(lyValue*) * initCapacity);

	if (!keys || !values)
	{
		free(keys);
		free(values);
		free(pDict);
		return false;
	}

	pDict->keys		= keys;
	pDict->values	= values;
	pDict->count	= 0;
	pDict->capacity = initCapacity;
	*ppDict			= pDict;

	return true;
}

void lyDestroyDict(lyDict* pDict)
{
	if (!pDict)
	{
		return;
	}

	for (size_t i = 0; i < pDict->count; i++)
	{
		free(pDict->keys[i]);
		lyDestroyValue(pDict->values[i]);
	}

	free(pDict->keys);
	free(pDict->values);
	free(pDict);
}

bool lyDictSetValue(lyDict* pDict, const char* key, lyValue* pValue)
{
	if (!pDict || !key || !pValue)
	{
		return false;
	}

	for (size_t i = 0; i < pDict->count; i++)
	{
		if (strcmp(pDict->keys[i], key) == 0)
		{
			lyDestroyValue(pDict->values[i]);
			pDict->values[i] = pValue;
			return true;
		}
	}

	if (pDict->count == pDict->capacity)
	{
		size_t	  newCapacity = pDict->capacity * 2;
		char**	  newKeys	  = (char**)malloc(sizeof(char*) * newCapacity);
		lyValue** newValues	  = (lyValue**)malloc(sizeof(lyValue*) * newCapacity);

		if (!newKeys || !newValues)
		{
			free(newKeys);
			free(newValues);
			return false;
		}

		memcpy(newKeys, pDict->keys, sizeof(char*) * pDict->count);
		memcpy(newValues, pDict->values, sizeof(lyValue*) * pDict->count);

		free(pDict->keys);
		free(pDict->values);

		pDict->keys		= newKeys;
		pDict->values	= newValues;
		pDict->capacity = newCapacity;
	}

	size_t keyLen  = strlen(key) + 1;
	char*  keyCopy = (char*)malloc(keyLen);
	if (!keyCopy)
	{
		return false;
	}

	memcpy(keyCopy, key, keyLen);

	pDict->keys[pDict->count]	= keyCopy;
	pDict->values[pDict->count] = pValue;
	pDict->count++;

	return true;
}

bool lyDictGetValue(lyValue** ppValue, const lyDict* pDict, const char* key)
{
	if (!ppValue || !pDict || !key)
	{
		return false;
	}

	for (size_t i = 0; i < pDict->count; i++)
	{
		if (strcmp(pDict->keys[i], key) == 0)
		{
			*ppValue = pDict->values[i];
			return true;
		}
	}

	return false;
}

void lyPrintDict(const lyDict* pDict)
{
	printf("Dict %p:\n", (const void*)pDict);
	for (size_t i = 0; i < pDict->count; i++)
	{
		printf("  %s: ", pDict->keys[i]);
		lyPrintValue(pDict->values[i]);
		printf("\n");
	}
}