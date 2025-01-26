#pragma once

#include "lyModel.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct lyZipEntry
{
	char*  filename;
	size_t offset;
	size_t size;
	bool   isCompressed;
} lyZipEntry;

typedef struct lyZipFile
{
	lyMappedFile mapping;
	lyZipEntry*	 entries;
	size_t		 entryCount;
} lyZipFile;

void lyZipOpen(lyZipFile** ppZip, const char* filename);
void lyZipClose(lyZipFile* pZip);

void lyZipFindEntry(lyZipEntry** ppEntry, const lyZipFile* pZip, const char* filename);
void lyZipFindEntryPattern(lyZipEntry** ppEntry, const lyZipFile* pZip, const char* pattern);

void lyZipGetEntryData(const uint8_t** ppData, const lyZipFile* pZip, const lyZipEntry* pEntry);