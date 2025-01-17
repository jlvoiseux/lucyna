#pragma once

#include "lyValue.h"

typedef struct lyStack
{
	lyValue** items;
	size_t	  count;
	size_t	  capacity;
} lyStack;

bool lyCreateStack(lyStack** ppStack, size_t initCapacity);
void lyDestroyStack(lyStack* pStack);

bool lyStackPush(lyStack* pStack, lyValue* pValue);
bool lyStackPop(lyValue** ppValue, lyStack* pStack);
bool lyStackPeek(lyValue** ppValue, const lyStack* pStack);