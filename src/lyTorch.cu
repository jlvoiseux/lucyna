#include "lyTorch.h"

#include <stdio.h>
#include <string.h>

void* lyTorchFindClass(const char* module, const char* name)
{
	if (!module || !name)
		return NULL;

	if (strcmp(module, "collections") == 0 && strcmp(name, "OrderedDict") == 0)
		return (void*)LY_TORCH_TYPE_TO_ID(LY_TORCH_ORDERED_DICT);

	if (strncmp(module, "torch", 5) != 0)
		return NULL;

	if (strcmp(module, "torch._utils") == 0 && strcmp(name, "_rebuild_tensor_v2") == 0)
		return (void*)LY_TORCH_TYPE_TO_ID(LY_TORCH_REBUILD_TENSOR);

	if (strcmp(module, "torch") == 0 && strcmp(name, "BFloat16Storage") == 0)
		return (void*)LY_TORCH_TYPE_TO_ID(LY_TORCH_BFLOAT16_STORAGE);

	return NULL;
}

void* lyTorchPersistentLoad(lyPickleReader* pReader, void** pidArray, size_t pidArraySize)
{
	if (!pReader || !pidArray || pidArraySize < 5)
		return NULL;

	lyValue* firstVal = (lyValue*)pidArray[0];
	void*	 storageStr;
	lyValueGetPtr(firstVal, &storageStr);
	if (strcmp((const char*)storageStr, "storage") != 0)
		return NULL;

	lyTorchType storageType = LY_TORCH_ID_TO_TYPE((lyTorchTypeId)pidArray[1]);

	lyValue* filenameVal = (lyValue*)pidArray[2];
	void*	 filenameStem;
	lyValueGetPtr(filenameVal, &filenameStem);

	lyValue* countVal = (lyValue*)pidArray[4];
	int64_t	 count;
	lyValueGetInt(countVal, &count);

	char filename[1024];
	snprintf(filename, sizeof(filename), "consolidated.00/data/%s", (const char*)filenameStem);

	lyZipFile*		pZip = (lyZipFile*)pReader->context;
	lyTorchStorage* pStorage;
	lyTorchCreateStorage(&pStorage, filename, storageType, 0, (size_t)count, pZip);

	return pStorage;
}

void lyTorchRebuildTensor(lyTensor** ppTensor, lyValue** args, size_t argCount)
{
	void*	storagePtr;
	int64_t storageOffset;
	void*	shapeArrayPtr;

	lyValueGetPtr(args[0], &storagePtr);
	lyTorchStorage* storage = (lyTorchStorage*)storagePtr;
	lyValueGetInt(args[1], &storageOffset);
	lyValueGetPtr(args[2], &shapeArrayPtr);
	lyStack* shapeArray = (lyStack*)shapeArrayPtr;

	int32_t* shape		   = (int32_t*)malloc(sizeof(int32_t) * shapeArray->count);
	size_t	 totalElements = 1;
	for (size_t i = 0; i < shapeArray->count; i++)
	{
		int64_t val;
		lyValueGetInt(shapeArray->items[i], &val);
		shape[i] = (int32_t)val;
		totalElements *= shape[i];
	}

	lyTensor* pTensor;
	lyTensorCreate(&pTensor, shape, (int32_t)shapeArray->count, NULL, NULL);
	free(shape);

	nv_bfloat16* srcData = (nv_bfloat16*)((uint8_t*)storage->rawData + storageOffset);
	for (size_t i = 0; i < totalElements; i++)
	{
		pTensor->data[i] = srcData[i];
	}

	*ppTensor = pTensor;
}

void lyTorchCreateStorage(lyTorchStorage** ppStorage, const char* filename, lyTorchType storageType, size_t offset, size_t elementCount, lyZipFile* pZip)
{
	lyTorchStorage* pStorage	 = (lyTorchStorage*)malloc(sizeof(lyTorchStorage));
	size_t			filenameLen	 = strlen(filename) + 1;
	char*			filenameCopy = (char*)malloc(filenameLen);
	memcpy(filenameCopy, filename, filenameLen);

	lyZipEntry*	   entry;
	const uint8_t* data;

	lyZipFindEntry(&entry, pZip, filename);
	lyZipGetEntryData(&data, pZip, entry);

	pStorage->filename		= filenameCopy;
	pStorage->storageType	= storageType;
	pStorage->storageOffset = offset;
	pStorage->elementCount	= elementCount;
	pStorage->rawData		= (void*)data;

	*ppStorage = pStorage;
}

void lyTorchInitHandler(lyPickleReader* pReader)
{
	pReader->findClassFn	  = lyTorchFindClass;
	pReader->persistentLoadFn = lyTorchPersistentLoad;
}