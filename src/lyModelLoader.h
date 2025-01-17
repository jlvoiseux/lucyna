#pragma once

#include "lyModel.h"

#include <cuda_runtime.h>
#include <stdint.h>

bool lyMapModelFile(const char* filename, lyMappedFile* pMapping);
void lyUnmapModelFile(lyMappedFile* pMapping);

bool lyLoadModelArgs(lyModel* pModel, const char* modelDir);

bool lyLoadModel(lyModel** ppModel, const char* modelDir, bool includeTensors, bool includeVocab);
void lyDestroyModel(lyModel* pModel);