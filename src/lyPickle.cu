#include "lyPickle.h"
#include "lyTorch.h"

#include <stdlib.h>
#include <string.h>

static bool popMark(lyStack** ppItems, lyPickleReader* pReader)
{
	if (!ppItems || !pReader || !pReader->stack || !pReader->metastack)
	{
		return false;
	}

	*ppItems = pReader->stack;

	lyValue* prevStackVal;
	if (!lyStackPop(&prevStackVal, pReader->metastack))
	{
		return false;
	}

	void* prevStackPtr;
	if (!lyGetPtrValue(prevStackVal, &prevStackPtr))
	{
		lyDestroyValue(prevStackVal);
		return false;
	}
	lyDestroyValue(prevStackVal);

	pReader->stack = (lyStack*)prevStackPtr;
	return true;
}

bool lyCreatePickleReader(lyPickleReader** ppReader, const uint8_t* data, size_t size)
{
	if (!ppReader || !data)
	{
		return false;
	}

	lyPickleReader* pReader = (lyPickleReader*)malloc(sizeof(lyPickleReader));
	if (!pReader)
	{
		return false;
	}

	if (!lyCreateStack(&pReader->stack, 16) || !lyCreateStack(&pReader->metastack, 4) || !lyCreateDict(&pReader->memo, 32))
	{
		lyDestroyPickleReader(pReader);
		return false;
	}

	pReader->data			  = data;
	pReader->size			  = size;
	pReader->pos			  = 0;
	pReader->proto			  = 0;
	pReader->findClassFn	  = NULL;
	pReader->persistentLoadFn = NULL;
	pReader->context		  = NULL;

	*ppReader = pReader;
	return true;
}

void lyDestroyPickleReader(lyPickleReader* pReader)
{
	if (!pReader)
	{
		return;
	}

	lyDestroyStack(pReader->stack);
	lyDestroyStack(pReader->metastack);
	lyDestroyDict(pReader->memo);
	free(pReader);
}

static uint8_t readByte(lyPickleReader* pReader)
{
	if (pReader->pos >= pReader->size)
		return 0;
	return pReader->data[pReader->pos++];
}

static bool readUInt16(uint16_t* pValue, lyPickleReader* pReader)
{
	if (!pValue || !pReader)
		return false;

	*pValue = readByte(pReader);
	*pValue |= (uint16_t)readByte(pReader) << 8;
	return true;
}

static bool readUInt32(uint32_t* pValue, lyPickleReader* pReader)
{
	if (!pValue || !pReader)
		return false;

	*pValue = readByte(pReader);
	*pValue |= (uint32_t)readByte(pReader) << 8;
	*pValue |= (uint32_t)readByte(pReader) << 16;
	*pValue |= (uint32_t)readByte(pReader) << 24;
	return true;
}

static bool readLine(char** ppStr, lyPickleReader* pReader)
{
	if (!ppStr || !pReader)
	{
		return false;
	}

	const uint8_t* start = pReader->data + pReader->pos;
	const uint8_t* p	 = start;
	while (p < pReader->data + pReader->size && *p != '\n')
	{
		p++;
	}

	size_t len = p - start;
	char*  str = (char*)malloc(len + 1);
	if (!str)
	{
		return false;
	}

	memcpy(str, start, len);
	str[len] = '\0';

	pReader->pos += len + 1;
	*ppStr = str;
	return true;
}

static bool readString(char** ppStr, lyPickleReader* pReader, uint32_t length)
{
	if (!ppStr || !pReader || pReader->pos + length > pReader->size)
	{
		return false;
	}

	char* str = (char*)malloc(length + 1);
	if (!str)
	{
		return false;
	}

	memcpy(str, pReader->data + pReader->pos, length);
	str[length] = '\0';

	pReader->pos += length;
	*ppStr = str;
	return true;
}

static bool load_PROTO(lyPickleReader* pReader)
{
	uint8_t proto = readByte(pReader);
	if (proto > 5)
		return false;
	pReader->proto = proto;
	return true;
}

static bool load_EMPTY_DICT(lyPickleReader* pReader)
{
	lyDict* dict;
	if (!lyCreateDict(&dict, 8))
	{
		return false;
	}

	lyValue* value;
	if (!lyCreatePtrValue(&value, dict))
	{
		lyDestroyDict(dict);
		return false;
	}

	if (!lyStackPush(pReader->stack, value))
	{
		lyDestroyValue(value);
		return false;
	}

	return true;
}

static bool load_MARK(lyPickleReader* pReader)
{
	lyStack* newStack;
	if (!lyCreateStack(&newStack, 8))
		return false;

	lyValue* stackValue;
	if (!lyCreatePtrValue(&stackValue, pReader->stack))
	{
		lyDestroyStack(newStack);
		return false;
	}

	if (!lyStackPush(pReader->metastack, stackValue))
	{
		lyDestroyValue(stackValue);
		lyDestroyStack(newStack);
		return false;
	}

	pReader->stack = newStack;
	return true;
}

static bool load_GLOBAL(lyPickleReader* pReader)
{
	char* module;
	if (!readLine(&module, pReader))
		return false;

	char* name;
	if (!readLine(&name, pReader))
	{
		free(module);
		return false;
	}

	if (pReader->findClassFn)
	{
		void* obj = pReader->findClassFn(module, name);
		if (obj)
		{
			lyValue* value;
			if (!lyCreatePtrValue(&value, obj))
			{
				free(module);
				free(name);
				return false;
			}

			if (!lyStackPush(pReader->stack, value))
			{
				lyDestroyValue(value);
				free(module);
				free(name);
				return false;
			}
		}
	}

	free(module);
	free(name);
	return true;
}

static bool load_REDUCE(lyPickleReader* pReader)
{
	lyValue* argsVal;
	if (!lyStackPop(&argsVal, pReader->stack))
		return false;

	lyValue* funcVal;
	if (!lyStackPop(&funcVal, pReader->stack))
	{
		lyDestroyValue(argsVal);
		return false;
	}

	void *argsPtr, *funcPtr;
	if (!lyGetPtrValue(argsVal, &argsPtr) || !lyGetPtrValue(funcVal, &funcPtr))
	{
		lyDestroyValue(argsVal);
		lyDestroyValue(funcVal);
		return false;
	}

	lyTorchTypeId funcType = (lyTorchTypeId)funcPtr;
	lyStack*	  args	   = (lyStack*)argsPtr;

	void*	  result = NULL;
	lyTensor* pTensor;
	switch (LY_TORCH_ID_TO_TYPE(funcType))
	{
		case LY_TORCH_REBUILD_TENSOR:
			if (!lyRebuildTensor(&pTensor, args->items, args->count))
			{
				lyDestroyValue(argsVal);
				lyDestroyValue(funcVal);
				return false;
			}
			result = pTensor;
			break;

		case LY_TORCH_ORDERED_DICT:
			if (!lyCreateDict((lyDict**)&result, 8))
			{
				lyDestroyValue(argsVal);
				lyDestroyValue(funcVal);
				return false;
			}
			break;

		default:
			lyDestroyValue(argsVal);
			lyDestroyValue(funcVal);
			return false;
	}

	lyValue* resultValue;
	if (!lyCreatePtrValue(&resultValue, result))
	{
		if (LY_TORCH_ID_TO_TYPE(funcType) == LY_TORCH_ORDERED_DICT)
			lyDestroyDict((lyDict*)result);
		else
			lyDestroyTensor((lyTensor*)result);

		lyDestroyValue(argsVal);
		lyDestroyValue(funcVal);
		return false;
	}

	if (!lyStackPush(pReader->stack, resultValue))
	{
		lyDestroyValue(resultValue);
		lyDestroyValue(argsVal);
		lyDestroyValue(funcVal);
		return false;
	}

	lyDestroyValue(argsVal);
	lyDestroyValue(funcVal);
	return true;
}

static bool load_STOP(lyPickleReader* pReader)
{
	return false;
}

static bool load_BINPERSID(lyPickleReader* pReader)
{
	lyValue* pidArrayVal;
	if (!lyStackPop(&pidArrayVal, pReader->stack))
		return false;

	lyStack* pidArray;
	if (!lyGetPtrValue(pidArrayVal, (void**)&pidArray))
	{
		lyDestroyValue(pidArrayVal);
		return false;
	}

	void* result = NULL;
	if (pReader->persistentLoadFn)
	{
		result = pReader->persistentLoadFn(pReader, (void**)pidArray->items, pidArray->count);
	}
	lyDestroyValue(pidArrayVal);

	if (!result)
		return false;

	lyValue* resultValue;
	if (!lyCreatePtrValue(&resultValue, result))
		return false;

	if (!lyStackPush(pReader->stack, resultValue))
	{
		lyDestroyValue(resultValue);
		return false;
	}

	return true;
}

static bool load_BINUNICODE(lyPickleReader* pReader)
{
	uint32_t length;
	if (!readUInt32(&length, pReader))
		return false;

	char* str;
	if (!readString(&str, pReader, length))
		return false;

	lyValue* value;
	if (!lyCreatePtrValue(&value, str))
	{
		free(str);
		return false;
	}

	if (!lyStackPush(pReader->stack, value))
	{
		lyDestroyValue(value);
		return false;
	}

	return true;
}

static bool load_BINPUT(lyPickleReader* pReader)
{
	uint8_t idx = readByte(pReader);

	lyValue* value;
	if (!lyStackPeek(&value, pReader->stack))
		return false;

	lyValue* valueCopy;
	if (!lyCloneValue(&valueCopy, value))
		return false;

	char key[16];
	snprintf(key, sizeof(key), "%d", idx);

	if (!lyDictSetValue(pReader->memo, key, valueCopy))
	{
		lyDestroyValue(valueCopy);
		return false;
	}

	return true;
}

static bool load_LONG_BINPUT(lyPickleReader* pReader)
{
	uint32_t idx;
	if (!readUInt32(&idx, pReader))
		return false;

	lyValue* value;
	if (!lyStackPeek(&value, pReader->stack))
		return false;

	lyValue* valueCopy;
	if (!lyCloneValue(&valueCopy, value))
		return false;

	char key[16];
	snprintf(key, sizeof(key), "%u", idx);

	if (!lyDictSetValue(pReader->memo, key, valueCopy))
	{
		lyDestroyValue(valueCopy);
		return false;
	}

	return true;
}

static bool load_BINSTRING(lyPickleReader* pReader)
{
	uint32_t length;
	if (!readUInt32(&length, pReader))
		return false;

	char* str;
	if (!readString(&str, pReader, length))
		return false;

	lyValue* value;
	if (!lyCreatePtrValue(&value, str))
	{
		free(str);
		return false;
	}

	if (!lyStackPush(pReader->stack, value))
	{
		lyDestroyValue(value);
		return false;
	}

	return true;
}

static bool load_SHORT_BINSTRING(lyPickleReader* pReader)
{
	uint8_t length = readByte(pReader);

	char* str;
	if (!readString(&str, pReader, length))
		return false;

	lyValue* value;
	if (!lyCreatePtrValue(&value, str))
	{
		free(str);
		return false;
	}

	if (!lyStackPush(pReader->stack, value))
	{
		lyDestroyValue(value);
		return false;
	}

	return true;
}

static bool load_BININT(lyPickleReader* pReader)
{
	uint32_t value;
	if (!readUInt32(&value, pReader))
		return false;

	lyValue* intValue;
	if (!lyCreateIntValue(&intValue, (int32_t)value))
		return false;

	if (!lyStackPush(pReader->stack, intValue))
	{
		lyDestroyValue(intValue);
		return false;
	}

	return true;
}

static bool load_BININT1(lyPickleReader* pReader)
{
	uint8_t val = readByte(pReader);

	lyValue* value;
	if (!lyCreateIntValue(&value, val))
		return false;

	if (!lyStackPush(pReader->stack, value))
	{
		lyDestroyValue(value);
		return false;
	}

	return true;
}

static bool load_BININT2(lyPickleReader* pReader)
{
	uint16_t val;
	if (!readUInt16(&val, pReader))
		return false;

	lyValue* value;
	if (!lyCreateIntValue(&value, val))
		return false;

	if (!lyStackPush(pReader->stack, value))
	{
		lyDestroyValue(value);
		return false;
	}

	return true;
}

static bool load_BINGET(lyPickleReader* pReader)
{
	uint8_t idx = readByte(pReader);

	char key[16];
	snprintf(key, sizeof(key), "%d", idx);

	lyValue* value;
	if (!lyDictGetValue(&value, pReader->memo, key))
		return false;

	lyValue* valueCopy;
	if (!lyCloneValue(&valueCopy, value))
		return false;

	if (!lyStackPush(pReader->stack, valueCopy))
	{
		lyDestroyValue(valueCopy);
		return false;
	}

	return true;
}

static bool load_TUPLE(lyPickleReader* pReader)
{
	lyStack* items;
	if (!popMark(&items, pReader))
		return false;

	lyStack* tuple;
	if (!lyCreateStack(&tuple, items->count))
	{
		lyDestroyStack(items);
		return false;
	}

	for (size_t i = 0; i < items->count; i++)
	{
		lyValue* valueCopy;
		if (!lyCloneValue(&valueCopy, items->items[i]))
		{
			lyDestroyStack(tuple);
			lyDestroyStack(items);
			return false;
		}

		if (!lyStackPush(tuple, valueCopy))
		{
			lyDestroyValue(valueCopy);
			lyDestroyStack(tuple);
			lyDestroyStack(items);
			return false;
		}
	}

	lyDestroyStack(items);

	lyValue* tupleValue;
	if (!lyCreatePtrValue(&tupleValue, tuple))
	{
		lyDestroyStack(tuple);
		return false;
	}

	if (!lyStackPush(pReader->stack, tupleValue))
	{
		lyDestroyValue(tupleValue);
		return false;
	}

	return true;
}

static bool load_TUPLE1(lyPickleReader* pReader)
{
	lyValue* val;
	if (!lyStackPop(&val, pReader->stack))
		return false;

	lyStack* tuple;
	if (!lyCreateStack(&tuple, 1))
	{
		lyDestroyValue(val);
		return false;
	}

	if (!lyStackPush(tuple, val))
	{
		lyDestroyValue(val);
		lyDestroyStack(tuple);
		return false;
	}

	lyValue* tupleValue;
	if (!lyCreatePtrValue(&tupleValue, tuple))
	{
		lyDestroyStack(tuple);
		return false;
	}

	if (!lyStackPush(pReader->stack, tupleValue))
	{
		lyDestroyValue(tupleValue);
		return false;
	}

	return true;
}

static bool load_TUPLE2(lyPickleReader* pReader)
{
	lyValue* val2;
	if (!lyStackPop(&val2, pReader->stack))
		return false;

	lyValue* val1;
	if (!lyStackPop(&val1, pReader->stack))
	{
		lyDestroyValue(val2);
		return false;
	}

	lyStack* tuple;
	if (!lyCreateStack(&tuple, 2))
	{
		lyDestroyValue(val1);
		lyDestroyValue(val2);
		return false;
	}

	if (!lyStackPush(tuple, val1) || !lyStackPush(tuple, val2))
	{
		lyDestroyValue(val1);
		lyDestroyValue(val2);
		lyDestroyStack(tuple);
		return false;
	}

	lyValue* tupleValue;
	if (!lyCreatePtrValue(&tupleValue, tuple))
	{
		lyDestroyStack(tuple);
		return false;
	}

	if (!lyStackPush(pReader->stack, tupleValue))
	{
		lyDestroyValue(tupleValue);
		return false;
	}

	return true;
}

static bool load_TUPLE3(lyPickleReader* pReader)
{
	lyValue* val3;
	if (!lyStackPop(&val3, pReader->stack))
		return false;

	lyValue* val2;
	if (!lyStackPop(&val2, pReader->stack))
	{
		lyDestroyValue(val3);
		return false;
	}

	lyValue* val1;
	if (!lyStackPop(&val1, pReader->stack))
	{
		lyDestroyValue(val2);
		lyDestroyValue(val3);
		return false;
	}

	lyStack* tuple;
	if (!lyCreateStack(&tuple, 3))
	{
		lyDestroyValue(val1);
		lyDestroyValue(val2);
		lyDestroyValue(val3);
		return false;
	}

	if (!lyStackPush(tuple, val1) || !lyStackPush(tuple, val2) || !lyStackPush(tuple, val3))
	{
		lyDestroyValue(val1);
		lyDestroyValue(val2);
		lyDestroyValue(val3);
		lyDestroyStack(tuple);
		return false;
	}

	lyValue* tupleValue;
	if (!lyCreatePtrValue(&tupleValue, tuple))
	{
		lyDestroyStack(tuple);
		return false;
	}

	if (!lyStackPush(pReader->stack, tupleValue))
	{
		lyDestroyValue(tupleValue);
		return false;
	}

	return true;
}

static bool load_EMPTY_TUPLE(lyPickleReader* pReader)
{
	lyStack* tuple;
	if (!lyCreateStack(&tuple, 1))
		return false;

	lyValue* tupleValue;
	if (!lyCreatePtrValue(&tupleValue, tuple))
	{
		lyDestroyStack(tuple);
		return false;
	}

	if (!lyStackPush(pReader->stack, tupleValue))
	{
		lyDestroyValue(tupleValue);
		return false;
	}

	return true;
}

static bool load_NEWTRUE(lyPickleReader* pReader)
{
	lyValue* value;
	if (!lyCreateBoolValue(&value, true))
		return false;

	if (!lyStackPush(pReader->stack, value))
	{
		lyDestroyValue(value);
		return false;
	}

	return true;
}

static bool load_NEWFALSE(lyPickleReader* pReader)
{
	lyValue* value;
	if (!lyCreateBoolValue(&value, false))
		return false;

	if (!lyStackPush(pReader->stack, value))
	{
		lyDestroyValue(value);
		return false;
	}

	return true;
}

static bool load_SETITEMS(lyPickleReader* pReader)
{
	lyStack* items;
	if (!popMark(&items, pReader))
		return false;

	lyValue* dictVal;
	if (!lyStackPeek(&dictVal, pReader->stack))
	{
		lyDestroyStack(items);
		return false;
	}

	void* dictPtr;
	if (!lyGetPtrValue(dictVal, &dictPtr))
	{
		lyDestroyStack(items);
		return false;
	}

	lyDict* dict = (lyDict*)dictPtr;

	for (size_t i = 0; i < items->count; i += 2)
	{
		void* keyPtr;
		if (!lyGetPtrValue(items->items[i], &keyPtr))
		{
			lyDestroyStack(items);
			return false;
		}

		if (!lyDictSetValue(dict, (const char*)keyPtr, items->items[i + 1]))
		{
			lyDestroyStack(items);
			return false;
		}

		items->items[i + 1] = NULL;	 // Ownership transferred
		lyDestroyValue(items->items[i]);
		items->items[i] = NULL;
	}

	lyDestroyStack(items);
	return true;
}

static bool dispatch(lyPickleReader* pReader, uint8_t opcode)
{
	static const struct
	{
		uint8_t opcode;
		bool (*handler)(lyPickleReader*);
	} DISPATCH_TABLE[] = {{LY_PROTO, load_PROTO},
						  {LY_EMPTY_DICT, load_EMPTY_DICT},
						  {LY_MARK, load_MARK},
						  {LY_GLOBAL, load_GLOBAL},
						  {LY_REDUCE, load_REDUCE},
						  {LY_STOP, load_STOP},
						  {LY_BINPERSID, load_BINPERSID},
						  {LY_BINUNICODE, load_BINUNICODE},
						  {LY_BINPUT, load_BINPUT},
						  {LY_LONG_BINPUT, load_LONG_BINPUT},
						  {LY_BINSTRING, load_BINSTRING},
						  {LY_SHORT_BINSTRING, load_SHORT_BINSTRING},
						  {LY_BININT, load_BININT},
						  {LY_BININT1, load_BININT1},
						  {LY_BININT2, load_BININT2},
						  {LY_BINGET, load_BINGET},
						  {LY_TUPLE, load_TUPLE},
						  {LY_TUPLE1, load_TUPLE1},
						  {LY_TUPLE2, load_TUPLE2},
						  {LY_TUPLE3, load_TUPLE3},
						  {LY_EMPTY_TUPLE, load_EMPTY_TUPLE},
						  {LY_NEWTRUE, load_NEWTRUE},
						  {LY_NEWFALSE, load_NEWFALSE},
						  {LY_SETITEMS, load_SETITEMS},
						  {0, NULL}};

	for (size_t i = 0; DISPATCH_TABLE[i].handler != NULL; i++)
	{
		if (DISPATCH_TABLE[i].opcode == opcode)
			return DISPATCH_TABLE[i].handler(pReader);
	}
	return false;
}

bool lyLoadPickle(lyDict** ppDict, lyPickleReader* pReader)
{
	if (!ppDict || !pReader)
		return false;

	while (pReader->pos < pReader->size)
	{
		uint8_t opcode = readByte(pReader);
		if (!dispatch(pReader, opcode))
		{
			if (opcode == LY_STOP)
			{
				lyValue* val;
				if (!lyStackPop(&val, pReader->stack))
					return false;

				void* dictPtr;
				if (!lyGetPtrValue(val, &dictPtr))
				{
					lyDestroyValue(val);
					return false;
				}

				*ppDict = (lyDict*)dictPtr;
				lyDestroyValue(val);
				return true;
			}
			return false;
		}
	}
	return false;
}