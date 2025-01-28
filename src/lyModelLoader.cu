#include "lyModelLoader.h"
#include "lyPickle.h"
#include "lyTorch.h"
#include "lyZip.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

void lyModelLoaderMapFile(const char* filename, lyMappedFile* pMapping)
{
#ifdef _WIN32
	pMapping->fileHandle = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

	LARGE_INTEGER fileSize;
	GetFileSizeEx(pMapping->fileHandle, &fileSize);
	pMapping->mappedSize	= (size_t)fileSize.QuadPart;
	pMapping->mappingHandle = CreateFileMappingA(pMapping->fileHandle, NULL, PAGE_READONLY, fileSize.HighPart, fileSize.LowPart, NULL);
	pMapping->mappedMemory	= MapViewOfFile(pMapping->mappingHandle, FILE_MAP_READ, 0, 0, 0);

#else
	pMapping->fd = open(filename, O_RDONLY);

	struct stat sb;
	fstat(pMapping->fd, &sb);
	pMapping->mappedSize   = sb.st_size;
	pMapping->mappedMemory = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, pMapping->fd, 0);
#endif
}

void lyModelLoaderUnmapFile(lyMappedFile* pMapping)
{
#ifdef _WIN32
	if (pMapping->mappedMemory)
	{
		UnmapViewOfFile(pMapping->mappedMemory);
	}
	if (pMapping->mappingHandle)
	{
		CloseHandle(pMapping->mappingHandle);
	}
	if (pMapping->fileHandle)
	{
		CloseHandle(pMapping->fileHandle);
	}
#else
	if (pMapping->mappedMemory)
	{
		munmap(pMapping->mappedMemory, pMapping->mappedSize);
	}
	if (pMapping->fd >= 0)
	{
		close(pMapping->fd);
	}
#endif
}

void lyModelLoaderLoadArgs(lyModel* pModel, const char* modelDir)
{
	char configPath[1024];
#ifdef _WIN32
	sprintf_s(configPath, sizeof(configPath), "%s\\params.json", modelDir);
#else
	snprintf(configPath, sizeof(configPath), "%s/params.json", modelDir);
#endif

	FILE* configFile = fopen(configPath, "r");
	char  line[1024];
	while (fgets(line, sizeof(line), configFile))
	{
		char* ptr = strchr(line, ':');
		if (!ptr)
		{
			continue;
		}

		*ptr = '\0';
		ptr++;

		char* key = line;
		while (*key == ' ' || *key == '\t' || *key == '"')
			key++;
		char* end = key + strlen(key) - 1;
		while (end > key && (*end == ' ' || *end == '\t' || *end == '"' || *end == '\n' || *end == '\r'))
			*end-- = '\0';

		char* value = ptr;
		while (*value == ' ' || *value == '\t' || *value == '"')
			value++;
		end = value + strlen(value) - 1;
		while (end > value && (*end == ' ' || *end == '\t' || *end == '"' || *end == '\n' || *end == '\r' || *end == ','))
			*end-- = '\0';

		if (strcmp(key, "dim") == 0)
			pModel->args.dim = atoi(value);
		else if (strcmp(key, "n_layers") == 0)
			pModel->args.nLayers = atoi(value);
		else if (strcmp(key, "n_heads") == 0)
			pModel->args.nHeads = atoi(value);
		else if (strcmp(key, "n_kv_heads") == 0)
			pModel->args.nKVHeads = atoi(value);
		else if (strcmp(key, "vocab_size") == 0)
			pModel->args.vocabSize = atoi(value);
		else if (strcmp(key, "multiple_of") == 0)
			pModel->args.multipleOf = atoi(value);
		else if (strcmp(key, "ffn_dim_multiplier") == 0)
			pModel->args.ffnDimMultiplier = (float)atof(value);
		else if (strcmp(key, "norm_eps") == 0)
			pModel->args.normEps = (float)atof(value);
		else if (strcmp(key, "rope_theta") == 0)
			pModel->args.ropeTheta = atof(value);
		else if (strcmp(key, "use_scaled_rope") == 0)
			pModel->args.useScaledRope = (strcmp(value, "true") == 0);
	}

	pModel->args.headDim = pModel->args.dim / pModel->args.nHeads;
	pModel->args.nRep	 = pModel->args.nHeads / pModel->args.nKVHeads;

	fclose(configFile);
}

void lyModelLoaderLoadModel(lyModel** ppModel, const char* modelDir)
{
	lyModel* pModel = (lyModel*)malloc(sizeof(lyModel));
	memset(pModel, 0, sizeof(lyModel));
	lyModelCreateDefaultArgs(&pModel->args);
	lyModelLoaderLoadArgs(pModel, modelDir);

	char modelPath[1024];
#ifdef _WIN32
	sprintf_s(modelPath, sizeof(modelPath), "%s\\consolidated.00.pth", modelDir);
#else
	snprintf(modelPath, sizeof(modelPath), "%s/consolidated.00.pth", modelDir);
#endif

	lyZipFile* pZip;
	lyZipOpen(&pZip, modelPath);

	lyZipEntry* pklEntry;
	lyZipFindEntryPattern(&pklEntry, pZip, "*.pkl");

	const uint8_t* pklData;
	lyZipGetEntryData(&pklData, pZip, pklEntry);

	lyPickleReader* pReader;
	lyPickleCreateReader(&pReader, pklData, pklEntry->size);

	lyTorchInitHandler(pReader);
	pReader->context = pZip;

	lyDict* tensors;
	lyPickleLoad(&tensors, pReader);

	pModel->tensorCount = tensors->count;
	pModel->tensors		= (lyTensor**)malloc(sizeof(lyTensor*) * tensors->count);
	memset(pModel->tensors, 0, sizeof(lyTensor*) * tensors->count);

	for (size_t i = 0; i < tensors->count; i++)
	{
		lyTensor* srcTensor;
		lyValueGetPtr(tensors->values[i], (void**)&srcTensor);
		lyTensorCreate(&pModel->tensors[i], srcTensor->shape, srcTensor->rank, srcTensor->data, tensors->keys[i]);
		free(srcTensor->data);
	}

	lyDictDestroy(tensors);
	lyPickleDestroyReader(pReader);
	lyZipClose(pZip);

	*ppModel = pModel;
}

void lyModelLoaderDestroyModel(lyModel* pModel)
{
	if (!pModel)
	{
		return;
	}

	if (pModel->tensors)
	{
		for (int32_t i = 0; i < pModel->tensorCount; i++)
		{
			lyTensorDestroy(pModel->tensors[i]);
		}
	}

	lyModelLoaderUnmapFile(&pModel->mapping);
	free(pModel);
}