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

bool lyOpenZip(lyZipFile** ppZip, const char* filename);
void lyCloseZip(lyZipFile* pZip);

bool lyFindZipEntry(lyZipEntry** ppEntry, const lyZipFile* pZip, const char* filename);
bool lyFindZipEntryPattern(lyZipEntry** ppEntry, const lyZipFile* pZip, const char* pattern);

bool lyGetZipEntryData(const uint8_t** ppData, const lyZipFile* pZip, const lyZipEntry* pEntry);