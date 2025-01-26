#pragma once

#include "lyValue.h"

typedef struct lyStack
{
	lyValue** items;
	size_t	  count;
	size_t	  capacity;
} lyStack;

void lyStackCreate(lyStack** ppStack, size_t initCapacity);
void lyStackDestroy(lyStack* pStack);

void lyStackPush(lyStack* pStack, lyValue* pValue);
void lyStackPop(lyValue** ppValue, lyStack* pStack);
void lyStackPeek(lyValue** ppValue, const lyStack* pStack);