#include "lyStack.h"

#include <stdlib.h>
#include <string.h>

void lyStackCreate(lyStack** ppStack, size_t initCapacity)
{
	lyStack* pStack	 = (lyStack*)malloc(sizeof(lyStack));
	pStack->items	 = (lyValue**)malloc(sizeof(lyValue*) * initCapacity);
	pStack->count	 = 0;
	pStack->capacity = initCapacity;

	*ppStack = pStack;
}

void lyStackDestroy(lyStack* pStack)
{
	for (size_t i = 0; i < pStack->count; i++)
		lyValueDestroy(pStack->items[i]);

	free(pStack->items);
	free(pStack);
}

void lyStackPush(lyStack* pStack, lyValue* pValue)
{
	if (pStack->count == pStack->capacity)
	{
		size_t	  newCapacity = pStack->capacity * 2;
		lyValue** pNewItems	  = (lyValue**)malloc(sizeof(lyValue*) * newCapacity);
		memcpy(pNewItems, pStack->items, sizeof(lyValue*) * pStack->count);
		free(pStack->items);

		pStack->items	 = pNewItems;
		pStack->capacity = newCapacity;
	}

	pStack->items[pStack->count++] = pValue;
}

void lyStackPop(lyValue** ppValue, lyStack* pStack)
{
	*ppValue = pStack->items[--pStack->count];
}

void lyStackPeek(lyValue** ppValue, const lyStack* pStack)
{
	*ppValue = pStack->items[pStack->count - 1];
}