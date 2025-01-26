#pragma once

#include "lyPickle.h"
#include "lyTensor.h"
#include "lyZip.h"

typedef enum lyTorchType
{
	LY_TORCH_REBUILD_TENSOR	  = 1,	// Using 0 would trigger null checks
	LY_TORCH_BFLOAT16_STORAGE = 2,
	LY_TORCH_ORDERED_DICT	  = 3,
} lyTorchType;

typedef uintptr_t lyTorchTypeId;
#define LY_TORCH_TYPE_TO_ID(type) ((lyTorchTypeId)(type))
#define LY_TORCH_ID_TO_TYPE(id) ((lyTorchType)(id))

typedef struct lyTorchStorage
{
	const char* filename;
	lyTorchType storageType;
	size_t		storageOffset;
	size_t		elementCount;
	void*		rawData;
} lyTorchStorage;

void lyTorchInitHandler(lyPickleReader* pReader);

void* lyTorchFindClass(const char* module, const char* name);
void* lyTorchPersistentLoad(lyPickleReader* pReader, void** pidArray, size_t pidArraySize);

void lyTorchRebuildTensor(lyTensor** ppTensor, lyValue** args, size_t argCount);
void lyTorchCreateStorage(lyTorchStorage** ppStorage, const char* filename, lyTorchType storageType, size_t offset, size_t elementCount, lyZipFile* pZip);