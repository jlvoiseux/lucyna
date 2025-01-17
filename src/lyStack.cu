#include "lyStack.h"

#include <stdlib.h>
#include <string.h>

bool lyCreateStack(lyStack** ppStack, size_t initCapacity)
{
	if (!ppStack || initCapacity == 0)
	{
		return false;
	}

	lyStack* pStack = (lyStack*)malloc(sizeof(lyStack));
	if (!pStack)
	{
		return false;
	}

	pStack->items = (lyValue**)malloc(sizeof(lyValue*) * initCapacity);
	if (!pStack->items)
	{
		free(pStack);
		return false;
	}

	pStack->count	 = 0;
	pStack->capacity = initCapacity;

	*ppStack = pStack;
	return true;
}

void lyDestroyStack(lyStack* pStack)
{
	if (!pStack)
	{
		return;
	}

	for (size_t i = 0; i < pStack->count; i++)
	{
		lyDestroyValue(pStack->items[i]);
	}

	free(pStack->items);
	free(pStack);
}

bool lyStackPush(lyStack* pStack, lyValue* pValue)
{
	if (!pStack || !pValue)
	{
		return false;
	}

	if (pStack->count == pStack->capacity)
	{
		size_t	  newCapacity = pStack->capacity * 2;
		lyValue** pNewItems	  = (lyValue**)malloc(sizeof(lyValue*) * newCapacity);
		if (!pNewItems)
		{
			return false;
		}

		memcpy(pNewItems, pStack->items, sizeof(lyValue*) * pStack->count);
		free(pStack->items);

		pStack->items	 = pNewItems;
		pStack->capacity = newCapacity;
	}

	pStack->items[pStack->count++] = pValue;
	return true;
}

bool lyStackPop(lyValue** ppValue, lyStack* pStack)
{
	if (!ppValue || !pStack || pStack->count == 0)
	{
		return false;
	}

	*ppValue = pStack->items[--pStack->count];
	return true;
}

bool lyStackPeek(lyValue** ppValue, const lyStack* pStack)
{
	if (!ppValue || !pStack || pStack->count == 0)
	{
		return false;
	}

	*ppValue = pStack->items[pStack->count - 1];
	return true;
}