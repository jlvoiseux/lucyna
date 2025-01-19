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

bool lyMapModelFile(const char* filename, lyMappedFile* pMapping)
{
	if (!filename || !pMapping)
		return false;

#ifdef _WIN32
	pMapping->fileHandle = CreateFileA(filename, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (pMapping->fileHandle == INVALID_HANDLE_VALUE)
	{
		fprintf(stderr, "Error opening file: %s\n", filename);
		return false;
	}

	LARGE_INTEGER fileSize;
	if (!GetFileSizeEx(pMapping->fileHandle, &fileSize))
	{
		CloseHandle(pMapping->fileHandle);
		return false;
	}
	pMapping->mappedSize = (size_t)fileSize.QuadPart;

	pMapping->mappingHandle = CreateFileMappingA(pMapping->fileHandle, NULL, PAGE_READONLY, fileSize.HighPart, fileSize.LowPart, NULL);
	if (!pMapping->mappingHandle)
	{
		CloseHandle(pMapping->fileHandle);
		return false;
	}

	pMapping->mappedMemory = MapViewOfFile(pMapping->mappingHandle, FILE_MAP_READ, 0, 0, 0);
	if (!pMapping->mappedMemory)
	{
		CloseHandle(pMapping->mappingHandle);
		CloseHandle(pMapping->fileHandle);
		return false;
	}

#else
	pMapping->fd = open(filename, O_RDONLY);
	if (pMapping->fd < 0)
	{
		fprintf(stderr, "Error opening file: %s\n", filename);
		return false;
	}

	struct stat sb;
	if (fstat(pMapping->fd, &sb) < 0)
	{
		close(pMapping->fd);
		return false;
	}

	pMapping->mappedSize   = sb.st_size;
	pMapping->mappedMemory = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, pMapping->fd, 0);

	if (pMapping->mappedMemory == MAP_FAILED)
	{
		close(pMapping->fd);
		return false;
	}
#endif

	return true;
}

void lyUnmapModelFile(lyMappedFile* pMapping)
{
	if (!pMapping)
		return;

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

bool lyLoadModelArgs(lyModel* pModel, const char* modelDir)
{
	if (!pModel || !modelDir)
	{
		return false;
	}

	char configPath[1024];
#ifdef _WIN32
	if (sprintf_s(configPath, sizeof(configPath), "%s\\params.json", modelDir) < 0)
#else
	if (snprintf(configPath, sizeof(configPath), "%s/params.json", modelDir) < 0)
#endif
	{
		return false;
	}

	FILE* configFile = fopen(configPath, "r");
	if (!configFile)
	{
		return false;
	}

	char line[1024];
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
			pModel->args.ropeTheta = (float)atof(value);
		else if (strcmp(key, "use_scaled_rope") == 0)
			pModel->args.useScaledRope = (strcmp(value, "true") == 0);
	}

	fclose(configFile);
	return true;
}

bool lyLoadModel(lyModel** ppModel, const char* modelDir, bool includeTensors, bool includeVocab)
{
	if (!ppModel || !modelDir)
	{
		return false;
	}

	lyModel* pModel = (lyModel*)malloc(sizeof(lyModel));
	if (!pModel)
	{
		return false;
	}

	memset(pModel, 0, sizeof(lyModel));

	if (!lyCreateDefaultModelArgs(&pModel->args))
	{
		free(pModel);
		return false;
	}

	if (!includeTensors)
	{
		*ppModel = pModel;
		return true;
	}

	if (!lyLoadModelArgs(pModel, modelDir))
	{
		lyDestroyModel(pModel);
		return false;
	}

	char modelPath[1024];
#ifdef _WIN32
	if (sprintf_s(modelPath, sizeof(modelPath), "%s\\consolidated.00.pth", modelDir) < 0)
#else
	if (snprintf(modelPath, sizeof(modelPath), "%s/consolidated.00.pth", modelDir) < 0)
#endif
	{
		lyDestroyModel(pModel);
		return false;
	}

	lyZipFile* pZip;
	if (!lyOpenZip(&pZip, modelPath))
	{
		lyDestroyModel(pModel);
		return false;
	}

	lyZipEntry* pklEntry;
	if (!lyFindZipEntryPattern(&pklEntry, pZip, "*.pkl"))
	{
		lyCloseZip(pZip);
		lyDestroyModel(pModel);
		return false;
	}

	const uint8_t* pklData;
	if (!lyGetZipEntryData(&pklData, pZip, pklEntry))
	{
		lyCloseZip(pZip);
		lyDestroyModel(pModel);
		return false;
	}

	lyPickleReader* pReader;
	if (!lyCreatePickleReader(&pReader, pklData, pklEntry->size))
	{
		lyCloseZip(pZip);
		lyDestroyModel(pModel);
		return false;
	}

	lyInitTorchHandlers(pReader);
	pReader->context = pZip;

	lyDict* tensors;
	if (!lyLoadPickle(&tensors, pReader))
	{
		lyDestroyPickleReader(pReader);
		lyCloseZip(pZip);
		lyDestroyModel(pModel);
		return false;
	}

	pModel->tensorCount = tensors->count;
	pModel->tensors		= (lyTensor*)malloc(sizeof(lyTensor) * tensors->count);
	if (!pModel->tensors)
	{
		lyDestroyDict(tensors);
		lyDestroyPickleReader(pReader);
		lyCloseZip(pZip);
		lyDestroyModel(pModel);
		return false;
	}

	memset(pModel->tensors, 0, sizeof(lyTensor) * tensors->count);

	for (size_t i = 0; i < tensors->count; i++)
	{
		lyTensor* srcTensor;
		if (!lyGetPtrValue(tensors->values[i], (void**)&srcTensor))
		{
			lyDestroyDict(tensors);
			lyDestroyPickleReader(pReader);
			lyCloseZip(pZip);
			lyDestroyModel(pModel);
			return false;
		}

		if (!lySetTensorName(&pModel->tensors[i], tensors->keys[i]) || (srcTensor->rank > 0 && !lySetTensorShape(&pModel->tensors[i], srcTensor->shape, srcTensor->rank)) || !lySetTensorData(&pModel->tensors[i], srcTensor->data, srcTensor->dataSize))
		{
			lyDestroyDict(tensors);
			lyDestroyPickleReader(pReader);
			lyCloseZip(pZip);
			lyDestroyModel(pModel);
			return false;
		}
	}

	lyDestroyDict(tensors);
	lyDestroyPickleReader(pReader);
	lyCloseZip(pZip);

	//	printf("Model configuration:\n");
	//	printf("  Dimension: %d\n", pModel->args.dim);
	//	printf("  Layers: %d\n", pModel->args.nLayers);
	//	printf("  Attention heads: %d\n", pModel->args.nHeads);
	//	printf("  KV heads: %d\n", pModel->args.nKVHeads);
	//	printf("  Vocabulary size: %d\n", pModel->args.vocabSize);
	//	printf("  Multiple of: %d\n", pModel->args.multipleOf);
	//	printf("  FFN dim multiplier: %.1f\n", pModel->args.ffnDimMultiplier);
	//	printf("  Norm epsilon: %.1e\n", pModel->args.normEps);
	//	printf("  Use scaled rope: %s\n", pModel->args.useScaledRope ? "true" : "false");
	//	printf("  Rope theta: %.1f\n", pModel->args.ropeTheta);
	//	printf("  Max sequence length: %d\n", pModel->args.maxSequenceLength);
	//	printf("  N rep: %d\n", pModel->args.nRep);
	//	printf("  Head dimension: %d\n", pModel->args.headDim);
	//
	//	printf("\nTensors loaded: %d\n", pModel->tensorCount);
	//	for (size_t i = 0; i < pModel->tensorCount; i++)
	//	{
	//		lyTensor* tensor = &pModel->tensors[i];
	//
	//		char shapeStr[256] = "[";
	//		int	 pos		   = 1;
	//		for (int32_t d = 0; d < tensor->rank; d++)
	//
	//		{
	//			if (d == 0)
	//			{
	//				pos += snprintf(shapeStr + pos, sizeof(shapeStr) - pos, "%d", tensor->shape[d]);
	//			}
	//			else
	//			{
	//				pos += snprintf(shapeStr + pos, sizeof(shapeStr) - pos, ",%d", tensor->shape[d]);
	//			}
	//		}
	//		snprintf(shapeStr + pos, sizeof(shapeStr) - pos, "]");
	//
	//		printf("  %-40s shape: %-20s size: %zu bytes\n", tensor->name, shapeStr, tensor->dataSize);
	//	}

	*ppModel = pModel;
	return true;
}

void lyDestroyModel(lyModel* pModel)
{
	if (!pModel)
	{
		return;
	}

	if (pModel->tensors)
	{
		for (int32_t i = 0; i < pModel->tensorCount; i++)
		{
			lyDestroyTensor(&pModel->tensors[i]);
		}
		free(pModel->tensors);
	}

	lyUnmapModelFile(&pModel->mapping);
	free(pModel);
}