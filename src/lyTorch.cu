#include "lyTorch.h"

#include <stdio.h>
#include <string.h>

void* lyFindTorchClass(const char* module, const char* name)
{
	if (!module || !name)
	{
		return NULL;
	}

	if (strcmp(module, "collections") == 0 && strcmp(name, "OrderedDict") == 0)
	{
		return (void*)LY_TORCH_TYPE_TO_ID(LY_TORCH_ORDERED_DICT);
	}

	if (strncmp(module, "torch", 5) != 0)
	{
		return NULL;
	}

	if (strcmp(module, "torch._utils") == 0 && strcmp(name, "_rebuild_tensor_v2") == 0)
	{
		return (void*)LY_TORCH_TYPE_TO_ID(LY_TORCH_REBUILD_TENSOR);
	}

	if (strcmp(module, "torch") == 0 && strcmp(name, "BFloat16Storage") == 0)
	{
		return (void*)LY_TORCH_TYPE_TO_ID(LY_TORCH_BFLOAT16_STORAGE);
	}

	return NULL;
}

void* lyPersistentLoadTorch(lyPickleReader* pReader, void** pidArray, size_t pidArraySize)
{
	if (!pReader || !pidArray || pidArraySize < 5)
	{
		return NULL;
	}

	lyValue* firstVal = (lyValue*)pidArray[0];
	void*	 storageStr;
	if (!lyGetPtrValue(firstVal, &storageStr) || strcmp((const char*)storageStr, "storage") != 0)
	{
		return NULL;
	}

	lyTorchType storageType = LY_TORCH_ID_TO_TYPE((lyTorchTypeId)pidArray[1]);

	lyValue* filenameVal = (lyValue*)pidArray[2];
	void*	 filenameStem;
	if (!lyGetPtrValue(filenameVal, &filenameStem))
	{
		return NULL;
	}

	lyValue* countVal = (lyValue*)pidArray[4];
	int64_t	 count;
	if (!lyGetIntValue(countVal, &count))
	{
		return NULL;
	}

	char filename[1024];
	if (snprintf(filename, sizeof(filename), "consolidated.00/data/%s", (const char*)filenameStem) < 0)
	{
		return NULL;
	}

	lyZipFile*		pZip = (lyZipFile*)pReader->context;
	lyTorchStorage* pStorage;
	if (!lyCreateTorchStorage(&pStorage, filename, storageType, 0, (size_t)count, pZip))
	{
		return NULL;
	}

	return pStorage;
}

bool lyRebuildTensor(lyTensor** ppTensor, lyValue** args, size_t argCount)
{
	if (!ppTensor || !args || argCount < 6)
	{
		return false;
	}

	void* storagePtr;
	if (!lyGetPtrValue(args[0], &storagePtr))
	{
		return false;
	}

	lyTorchStorage* storage = (lyTorchStorage*)storagePtr;

	int64_t storageOffset;
	if (!lyGetIntValue(args[1], &storageOffset))
	{
		return false;
	}

	void* shapeArrayPtr;
	if (!lyGetPtrValue(args[2], &shapeArrayPtr))
	{
		return false;
	}

	lyStack* shapeArray = (lyStack*)shapeArrayPtr;

	lyTensor* pTensor;
	int32_t*  shape = (int32_t*)malloc(sizeof(int32_t) * shapeArray->count);
	for (size_t i = 0; i < shapeArray->count; i++)
	{
		int64_t val;
		if (!lyGetIntValue(shapeArray->items[i], &val))
		{
			free(shape);
			return false;
		}
		shape[i] = (int32_t)val;
	}
	lyCreateTensor(&pTensor, shape, (int32_t)shapeArray->count, (nv_bfloat16*)((uint8_t*)storage->rawData + storageOffset), NULL);
	free(shape);

	*ppTensor = pTensor;
	return true;
}

bool lyCreateTorchStorage(lyTorchStorage** ppStorage, const char* filename, lyTorchType storageType, size_t offset, size_t elementCount, lyZipFile* pZip)
{
	if (!ppStorage || !filename || !pZip)
	{
		return false;
	}

	lyTorchStorage* pStorage = (lyTorchStorage*)malloc(sizeof(lyTorchStorage));
	if (!pStorage)
	{
		return false;
	}

	size_t filenameLen	= strlen(filename) + 1;
	char*  filenameCopy = (char*)malloc(filenameLen);
	if (!filenameCopy)
	{
		free(pStorage);
		return false;
	}
	memcpy(filenameCopy, filename, filenameLen);

	lyZipEntry* entry;
	if (!lyFindZipEntry(&entry, pZip, filename) || entry->isCompressed)
	{
		free(filenameCopy);
		free(pStorage);
		return false;
	}

	const uint8_t* data;
	if (!lyGetZipEntryData(&data, pZip, entry))
	{
		free(filenameCopy);
		free(pStorage);
		return false;
	}

	pStorage->filename		= filenameCopy;
	pStorage->storageType	= storageType;
	pStorage->storageOffset = offset;
	pStorage->elementCount	= elementCount;
	pStorage->rawData		= (void*)data;

	*ppStorage = pStorage;
	return true;
}

void lyInitTorchHandlers(lyPickleReader* pReader)
{
	if (!pReader)
	{
		return;
	}

	pReader->findClassFn	  = lyFindTorchClass;
	pReader->persistentLoadFn = lyPersistentLoadTorch;
}