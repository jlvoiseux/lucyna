#include "lyModelLoader.h"
#include "lyZip.h"

#include <stdlib.h>
#include <string.h>

#define ZIP_LOCAL_HEADER_SIGNATURE 0x04034B50
#define ZIP_CENTRAL_DIR_SIGNATURE 0x02014B50
#define ZIP_END_CENTRAL_SIGNATURE 0x06054B50

#define ZIP_STORED_METHOD 0

#pragma pack(push, 1)
typedef struct
{
	uint32_t signature;
	uint16_t versionNeeded;
	uint16_t flags;
	uint16_t compressionMethod;
	uint16_t lastModTime;
	uint16_t lastModDate;
	uint32_t crc32;
	uint32_t compressedSize;
	uint32_t uncompressedSize;
	uint16_t filenameLength;
	uint16_t extraFieldLength;
} lyZipLocalHeader;

typedef struct
{
	uint32_t signature;
	uint16_t diskNumber;
	uint16_t centralDirDisk;
	uint16_t numEntriesThisDisk;
	uint16_t numEntriesTotal;
	uint32_t centralDirSize;
	uint32_t centralDirOffset;
	uint16_t commentLength;
} lyZipEndCentralDir;

typedef struct
{
	uint32_t signature;
	uint16_t versionMade;
	uint16_t versionNeeded;
	uint16_t flags;
	uint16_t compressionMethod;
	uint16_t lastModTime;
	uint16_t lastModDate;
	uint32_t crc32;
	uint32_t compressedSize;
	uint32_t uncompressedSize;
	uint16_t filenameLength;
	uint16_t extraFieldLength;
	uint16_t commentLength;
	uint16_t diskNumberStart;
	uint16_t internalAttrs;
	uint32_t externalAttrs;
	uint32_t localHeaderOffset;
} lyZipCentralDirEntry;
#pragma pack(pop)

static void lyReadZipEntries(lyZipFile* pZip)
{
	const uint8_t* data = (const uint8_t*)pZip->mapping.mappedMemory;
	size_t		   size = pZip->mapping.mappedSize;

	const uint8_t* p = data + size - sizeof(lyZipEndCentralDir);
	while (p > data)
	{
		if (*(uint32_t*)p == ZIP_END_CENTRAL_SIGNATURE)
			break;
		p--;
	}

	const lyZipEndCentralDir* endDir = (const lyZipEndCentralDir*)p;
	pZip->entries					 = (lyZipEntry*)calloc(endDir->numEntriesTotal, sizeof(lyZipEntry));
	pZip->entryCount				 = endDir->numEntriesTotal;
	const uint8_t* centralDirPtr	 = data + endDir->centralDirOffset;

	for (size_t i = 0; i < pZip->entryCount; i++)
	{
		const lyZipCentralDirEntry* dirEntry = (const lyZipCentralDirEntry*)centralDirPtr;
		if (dirEntry->compressionMethod != ZIP_STORED_METHOD)
		{
			pZip->entries[i].isCompressed = true;
			centralDirPtr += sizeof(lyZipCentralDirEntry) + dirEntry->filenameLength + dirEntry->extraFieldLength + dirEntry->commentLength;
			continue;
		}

		pZip->entries[i].filename = (char*)malloc(dirEntry->filenameLength + 1);
		memcpy(pZip->entries[i].filename, centralDirPtr + sizeof(lyZipCentralDirEntry), dirEntry->filenameLength);
		pZip->entries[i].filename[dirEntry->filenameLength] = '\0';

		const lyZipLocalHeader* localHeader = (const lyZipLocalHeader*)(data + dirEntry->localHeaderOffset);
		pZip->entries[i].offset				= dirEntry->localHeaderOffset + sizeof(lyZipLocalHeader) + localHeader->filenameLength + localHeader->extraFieldLength;
		pZip->entries[i].size				= dirEntry->uncompressedSize;

		centralDirPtr += sizeof(lyZipCentralDirEntry) + dirEntry->filenameLength + dirEntry->extraFieldLength + dirEntry->commentLength;
	}
}

void lyZipOpen(lyZipFile** ppZip, const char* filename)
{
	lyZipFile* pZip = (lyZipFile*)calloc(1, sizeof(lyZipFile));
	lyModelLoaderMapFile(filename, &pZip->mapping);

	lyReadZipEntries(pZip);
	*ppZip = pZip;
}

void lyZipClose(lyZipFile* pZip)
{
	for (size_t i = 0; i < pZip->entryCount; i++)
		free(pZip->entries[i].filename);

	free(pZip->entries);
	lyModelLoaderUnmapFile(&pZip->mapping);
	free(pZip);
}

void lyZipFindEntry(lyZipEntry** ppEntry, const lyZipFile* pZip, const char* filename)
{
	for (size_t i = 0; i < pZip->entryCount; i++)
	{
		if (strcmp(pZip->entries[i].filename, filename) == 0)
		{
			*ppEntry = &pZip->entries[i];
		}
	}
}

void lyZipFindEntryPattern(lyZipEntry** ppEntry, const lyZipFile* pZip, const char* pattern)
{
	for (size_t i = 0; i < pZip->entryCount; i++)
	{
		const char* name	   = pZip->entries[i].filename;
		const char* pat		   = pattern;
		const char* nameBackup = NULL;
		const char* patBackup  = NULL;

		while (*name)
		{
			if (*pat == '*')
			{
				patBackup  = ++pat;
				nameBackup = name;
			}
			else if (*pat == *name || *pat == '?')
			{
				pat++;
				name++;
			}
			else if (patBackup)
			{
				pat	 = patBackup;
				name = ++nameBackup;
			}
			else
				break;
		}

		while (*pat == '*')
			pat++;

		if (!*pat && !*name)
		{
			*ppEntry = &pZip->entries[i];
		}
	}
}

void lyZipGetEntryData(const uint8_t** ppData, const lyZipFile* pZip, const lyZipEntry* pEntry)
{
	*ppData = (const uint8_t*)pZip->mapping.mappedMemory + pEntry->offset;
}