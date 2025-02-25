#include "lyPickle.h"
#include "lyTorch.h"

#include <stdlib.h>
#include <string.h>

static void popMark(lyStack** ppItems, lyPickleReader* pReader)
{
	*ppItems = pReader->stack;

	lyValue* prevStackVal;
	lyStackPop(&prevStackVal, pReader->metastack);

	void* prevStackPtr;
	lyValueGetPtr(prevStackVal, &prevStackPtr);
	lyValueDestroy(prevStackVal);
	pReader->stack = (lyStack*)prevStackPtr;
}

void lyPickleCreateReader(lyPickleReader** ppReader, const uint8_t* data, size_t size)
{
	lyPickleReader* pReader = (lyPickleReader*)malloc(sizeof(lyPickleReader));
	lyStackCreate(&pReader->stack, 16);
	lyStackCreate(&pReader->metastack, 4);
	lyDictCreate(&pReader->memo, 32);

	pReader->data			  = data;
	pReader->size			  = size;
	pReader->pos			  = 0;
	pReader->proto			  = 0;
	pReader->findClassFn	  = NULL;
	pReader->persistentLoadFn = NULL;
	pReader->context		  = NULL;

	*ppReader = pReader;
}

void lyPickleDestroyReader(lyPickleReader* pReader)
{
	lyStackDestroy(pReader->stack);
	lyStackDestroy(pReader->metastack);
	lyDictDestroy(pReader->memo);
	free(pReader);
}

static uint8_t readByte(lyPickleReader* pReader)
{
	if (pReader->pos >= pReader->size)
		return 0;
	return pReader->data[pReader->pos++];
}

static void readUInt16(uint16_t* pValue, lyPickleReader* pReader)
{
	*pValue = readByte(pReader);
	*pValue |= (uint16_t)readByte(pReader) << 8;
}

static void readUInt32(uint32_t* pValue, lyPickleReader* pReader)
{
	*pValue = readByte(pReader);
	*pValue |= (uint32_t)readByte(pReader) << 8;
	*pValue |= (uint32_t)readByte(pReader) << 16;
	*pValue |= (uint32_t)readByte(pReader) << 24;
}

static void readLine(char** ppStr, lyPickleReader* pReader)
{
	const uint8_t* start = pReader->data + pReader->pos;
	const uint8_t* p	 = start;
	while (p < pReader->data + pReader->size && *p != '\n')
	{
		p++;
	}

	size_t len = p - start;
	char*  str = (char*)malloc(len + 1);
	memcpy(str, start, len);
	str[len] = '\0';

	pReader->pos += len + 1;
	*ppStr = str;
}

static void readString(char** ppStr, lyPickleReader* pReader, uint32_t length)
{
	char* str = (char*)malloc(length + 1);
	memcpy(str, pReader->data + pReader->pos, length);
	str[length] = '\0';

	pReader->pos += length;
	*ppStr = str;
}

static void load_PROTO(lyPickleReader* pReader)
{
	uint8_t proto  = readByte(pReader);
	pReader->proto = proto;
}

static void load_EMPTY_DICT(lyPickleReader* pReader)
{
	lyDict* dict;
	lyDictCreate(&dict, 8);

	lyValue* value;
	lyValueCreatePtr(&value, dict);
	lyStackPush(pReader->stack, value);
}

static void load_MARK(lyPickleReader* pReader)
{
	lyStack* newStack;
	lyStackCreate(&newStack, 8);

	lyValue* stackValue;
	lyValueCreatePtr(&stackValue, pReader->stack);

	lyStackPush(pReader->metastack, stackValue);
	pReader->stack = newStack;
}

static void load_GLOBAL(lyPickleReader* pReader)
{
	char* module;
	char* name;
	readLine(&module, pReader);
	readLine(&name, pReader);

	if (pReader->findClassFn)
	{
		void* obj = pReader->findClassFn(module, name);
		if (obj)
		{
			lyValue* value;
			lyValueCreatePtr(&value, obj);
			lyStackPush(pReader->stack, value);
		}
	}

	free(module);
	free(name);
}

static void load_REDUCE(lyPickleReader* pReader)
{
	lyValue* argsVal;
	lyValue* funcVal;
	lyStackPop(&argsVal, pReader->stack);
	lyStackPop(&funcVal, pReader->stack);

	void *argsPtr, *funcPtr;
	lyValueGetPtr(argsVal, &argsPtr);
	lyValueGetPtr(funcVal, &funcPtr);

	lyTorchTypeId funcType = (lyTorchTypeId)funcPtr;
	lyStack*	  args	   = (lyStack*)argsPtr;

	void*	  result = NULL;
	lyTensor* pTensor;
	switch (LY_TORCH_ID_TO_TYPE(funcType))
	{
		case LY_TORCH_REBUILD_TENSOR:
			lyTorchRebuildTensor(&pTensor, args->items, args->count);
			result = pTensor;
			break;
		case LY_TORCH_ORDERED_DICT:
			lyDictCreate((lyDict**)&result, 8);
			break;
		default:
			lyValueDestroy(argsVal);
			lyValueDestroy(funcVal);
			return;
	}

	lyValue* resultValue;
	lyValueCreatePtr(&resultValue, result);
	lyStackPush(pReader->stack, resultValue);
	lyValueDestroy(argsVal);
	lyValueDestroy(funcVal);
}

static void load_BINPERSID(lyPickleReader* pReader)
{
	lyStack* pidArray;
	lyValue* pidArrayVal;
	lyStackPop(&pidArrayVal, pReader->stack);
	lyValueGetPtr(pidArrayVal, (void**)&pidArray);

	void* result = NULL;
	if (pReader->persistentLoadFn)
	{
		result = pReader->persistentLoadFn(pReader, (void**)pidArray->items, pidArray->count);
	}
	lyValueDestroy(pidArrayVal);

	lyValue* resultValue;
	lyValueCreatePtr(&resultValue, result);
	lyStackPush(pReader->stack, resultValue);
}

static void load_BINUNICODE(lyPickleReader* pReader)
{
	uint32_t length;
	char*	 str;
	lyValue* value;

	readUInt32(&length, pReader);
	readString(&str, pReader, length);
	lyValueCreatePtr(&value, str);
	lyStackPush(pReader->stack, value);
}

static void load_BINPUT(lyPickleReader* pReader)
{
	uint8_t idx = readByte(pReader);

	lyValue* value;
	lyValue* valueCopy;
	char	 key[16];

	lyStackPeek(&value, pReader->stack);
	lyValueClone(&valueCopy, value);
	snprintf(key, sizeof(key), "%d", idx);
	lyDictSetValue(pReader->memo, key, valueCopy);
}

static void load_LONG_BINPUT(lyPickleReader* pReader)
{
	uint32_t idx;
	lyValue* value;
	char	 key[16];

	readUInt32(&idx, pReader);
	lyStackPeek(&value, pReader->stack);

	lyValue* valueCopy;
	lyValueClone(&valueCopy, value);
	snprintf(key, sizeof(key), "%u", idx);
	lyDictSetValue(pReader->memo, key, valueCopy);
}

static void load_BINSTRING(lyPickleReader* pReader)
{
	uint32_t length;
	char*	 str;

	readUInt32(&length, pReader);
	readString(&str, pReader, length);

	lyValue* value;
	lyValueCreatePtr(&value, str);
	lyStackPush(pReader->stack, value);
}

static void load_SHORT_BINSTRING(lyPickleReader* pReader)
{
	uint8_t	 length = readByte(pReader);
	char*	 str;
	lyValue* value;

	readString(&str, pReader, length);
	lyValueCreatePtr(&value, str);
	lyStackPush(pReader->stack, value);
}

static void load_BININT(lyPickleReader* pReader)
{
	uint32_t value;
	lyValue* intValue;

	readUInt32(&value, pReader);
	lyValueCreateInt(&intValue, (int32_t)value);
	lyStackPush(pReader->stack, intValue);
}

static void load_BININT1(lyPickleReader* pReader)
{
	uint8_t val = readByte(pReader);

	lyValue* value;
	lyValueCreateInt(&value, val);
	lyStackPush(pReader->stack, value);
}

static void load_BININT2(lyPickleReader* pReader)
{
	uint16_t val;
	lyValue* value;

	readUInt16(&val, pReader);
	lyValueCreateInt(&value, val);
	lyStackPush(pReader->stack, value);
}

static void load_BINGET(lyPickleReader* pReader)
{
	uint8_t idx = readByte(pReader);

	char	 key[16];
	lyValue* value;
	lyValue* valueCopy;

	snprintf(key, sizeof(key), "%d", idx);
	lyDictGetValue(&value, pReader->memo, key);
	lyValueClone(&valueCopy, value);
	lyStackPush(pReader->stack, valueCopy);
}

static void load_TUPLE(lyPickleReader* pReader)
{
	lyStack* items;
	lyStack* tuple;

	popMark(&items, pReader);
	lyStackCreate(&tuple, items->count);

	for (size_t i = 0; i < items->count; i++)
	{
		lyValue* valueCopy;
		lyValueClone(&valueCopy, items->items[i]);
		lyStackPush(tuple, valueCopy);
	}

	lyStackDestroy(items);

	lyValue* tupleValue;
	lyValueCreatePtr(&tupleValue, tuple);
	lyStackPush(pReader->stack, tupleValue);
}

static void load_TUPLE1(lyPickleReader* pReader)
{
	lyValue* val;
	lyStackPop(&val, pReader->stack);

	lyStack* tuple;
	lyStackCreate(&tuple, 1);
	lyStackPush(tuple, val);

	lyValue* tupleValue;
	lyValueCreatePtr(&tupleValue, tuple);
	lyStackPush(pReader->stack, tupleValue);
}

static void load_TUPLE2(lyPickleReader* pReader)
{
	lyValue* val2;
	lyValue* val1;
	lyStackPop(&val2, pReader->stack);
	lyStackPop(&val1, pReader->stack);

	lyStack* tuple;
	lyStackCreate(&tuple, 2);
	lyStackPush(tuple, val1);
	lyStackPush(tuple, val2);

	lyValue* tupleValue;
	lyValueCreatePtr(&tupleValue, tuple);
	lyStackPush(pReader->stack, tupleValue);
}

static void load_TUPLE3(lyPickleReader* pReader)
{
	lyValue* val3;
	lyValue* val2;
	lyValue* val1;
	lyStackPop(&val3, pReader->stack);
	lyStackPop(&val2, pReader->stack);
	lyStackPop(&val1, pReader->stack);

	lyStack* tuple;
	lyStackCreate(&tuple, 3);
	lyStackPush(tuple, val1);
	lyStackPush(tuple, val2);
	lyStackPush(tuple, val3);

	lyValue* tupleValue;
	lyValueCreatePtr(&tupleValue, tuple);
	lyStackPush(pReader->stack, tupleValue);
}

static void load_EMPTY_TUPLE(lyPickleReader* pReader)
{
	lyStack* tuple;
	lyStackCreate(&tuple, 1);

	lyValue* tupleValue;
	lyValueCreatePtr(&tupleValue, tuple);

	lyStackPush(pReader->stack, tupleValue);
}

static void load_NEWTRUE(lyPickleReader* pReader)
{
	lyValue* value;
	lyValueCreateBool(&value, true);

	lyStackPush(pReader->stack, value);
}

static void load_NEWFALSE(lyPickleReader* pReader)
{
	lyValue* value;
	lyValueCreateBool(&value, false);

	lyStackPush(pReader->stack, value);
}

static void load_SETITEMS(lyPickleReader* pReader)
{
	lyStack* items;
	lyValue* dictVal;
	void*	 dictPtr;

	popMark(&items, pReader);
	lyStackPeek(&dictVal, pReader->stack);
	lyValueGetPtr(dictVal, &dictPtr);
	lyDict* dict = (lyDict*)dictPtr;

	for (size_t i = 0; i < items->count; i += 2)
	{
		void* keyPtr;
		lyValueGetPtr(items->items[i], &keyPtr);
		lyDictSetValue(dict, (const char*)keyPtr, items->items[i + 1]);

		items->items[i + 1] = NULL;
		lyValueDestroy(items->items[i]);
		items->items[i] = NULL;
	}

	lyStackDestroy(items);
}

static bool dispatch(lyPickleReader* pReader, uint8_t opcode)
{
	static const struct
	{
		uint8_t opcode;
		void (*handler)(lyPickleReader*);
	} DISPATCH_TABLE[] = {{LY_PROTO, load_PROTO},
						  {LY_EMPTY_DICT, load_EMPTY_DICT},
						  {LY_MARK, load_MARK},
						  {LY_GLOBAL, load_GLOBAL},
						  {LY_REDUCE, load_REDUCE},
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
		{
			DISPATCH_TABLE[i].handler(pReader);
			return true;
		}
	}
	return false;
}

void lyPickleLoad(lyDict** ppDict, lyPickleReader* pReader)
{
	while (pReader->pos < pReader->size)
	{
		uint8_t opcode = readByte(pReader);
		if (!dispatch(pReader, opcode))
		{
			if (opcode == LY_STOP)
			{
				lyValue* val;
				void*	 dictPtr;

				lyStackPop(&val, pReader->stack);
				lyValueGetPtr(val, &dictPtr);

				*ppDict = (lyDict*)dictPtr;
				lyValueDestroy(val);
			}
		}
	}
}