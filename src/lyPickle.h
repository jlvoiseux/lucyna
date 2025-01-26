#pragma once

#include "lyDict.h"
#include "lyStack.h"
#include "lyValue.h"

#include <stdio.h>

typedef struct lyPickleReader
{
	const uint8_t* data;
	size_t		   size;
	size_t		   pos;
	uint8_t		   proto;
	lyStack*	   stack;
	lyStack*	   metastack;
	lyDict*		   memo;
	void* (*findClassFn)(const char* module, const char* name);
	void* (*persistentLoadFn)(struct lyPickleReader* pReader, void** pidArray, size_t pidArraySize);
	void* context;
} lyPickleReader;

#define LY_PROTO 0x80
#define LY_EMPTY_DICT '}'
#define LY_MARK '('
#define LY_GLOBAL 'c'
#define LY_REDUCE 'R'
#define LY_STOP '.'
#define LY_BINPERSID 'Q'
#define LY_BINUNICODE 'X'
#define LY_BINPUT 'q'
#define LY_LONG_BINPUT 'r'
#define LY_BINSTRING 'T'
#define LY_SHORT_BINSTRING 'U'
#define LY_BININT 'J'
#define LY_BININT1 'K'
#define LY_BININT2 'M'
#define LY_BINGET 'h'
#define LY_TUPLE 't'
#define LY_TUPLE1 0x85
#define LY_TUPLE2 0x86
#define LY_TUPLE3 0x87
#define LY_EMPTY_TUPLE ')'
#define LY_NEWTRUE 0x88
#define LY_NEWFALSE 0x89
#define LY_SETITEMS 'u'

void lyPickleCreateReader(lyPickleReader** ppReader, const uint8_t* data, size_t size);
void lyPickleDestroyReader(lyPickleReader* pReader);
void lyPickleLoad(lyDict** ppDict, lyPickleReader* pReader);