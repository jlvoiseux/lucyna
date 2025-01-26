#pragma once

#include "lyModel.h"

#include <cuda_runtime.h>
#include <stdint.h>

void lyModelLoaderMapFile(const char* filename, lyMappedFile* pMapping);
void lyModelLoaderUnmapFile(lyMappedFile* pMapping);

void lyModelLoaderLoadArgs(lyModel* pModel, const char* modelDir);

void lyModelLoaderLoadModel(lyModel** ppModel, const char* modelDir);
void lyModelLoaderDestroyModel(lyModel* pModel);