#include "lyDict.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void lyDictCreate(lyDict** ppDict, size_t initCapacity)
{
	lyDict*	  pDict	 = (lyDict*)malloc(sizeof(lyDict));
	char**	  keys	 = (char**)malloc(sizeof(char*) * initCapacity);
	lyValue** values = (lyValue**)malloc(sizeof(lyValue*) * initCapacity);

	pDict->keys		= keys;
	pDict->values	= values;
	pDict->count	= 0;
	pDict->capacity = initCapacity;
	*ppDict			= pDict;
}

void lyDictDestroy(lyDict* pDict)
{
	for (size_t i = 0; i < pDict->count; i++)
	{
		free(pDict->keys[i]);
		lyValueDestroy(pDict->values[i]);
	}

	free(pDict->keys);
	free(pDict->values);
	free(pDict);
}

void lyDictSetValue(lyDict* pDict, const char* key, lyValue* pValue)
{
	for (size_t i = 0; i < pDict->count; i++)
	{
		if (strcmp(pDict->keys[i], key) == 0)
		{
			lyValueDestroy(pDict->values[i]);
			pDict->values[i] = pValue;
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
	memcpy(keyCopy, key, keyLen);

	pDict->keys[pDict->count]	= keyCopy;
	pDict->values[pDict->count] = pValue;
	pDict->count++;
}

void lyDictGetValue(lyValue** ppValue, const lyDict* pDict, const char* key)
{
	for (size_t i = 0; i < pDict->count; i++)
	{
		if (strcmp(pDict->keys[i], key) == 0)
		{
			*ppValue = pDict->values[i];
		}
	}
}

void lyDictPrint(const lyDict* pDict)
{
	printf("Dict %p:\n", (const void*)pDict);
	for (size_t i = 0; i < pDict->count; i++)
	{
		printf("  %s: ", pDict->keys[i]);
		lyValuePrint(pDict->values[i]);
		printf("\n");
	}
}