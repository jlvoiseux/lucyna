#include "lyModel.h"

#include <stdlib.h>
#include <string.h>

bool lyCreateDefaultModelArgs(lyModelArgs* pArgs)
{
	if (!pArgs)
	{
		return false;
	}

	pArgs->dim				 = 4096;
	pArgs->nLayers			 = 32;
	pArgs->nHeads			 = 32;
	pArgs->nKVHeads			 = -1;
	pArgs->vocabSize		 = -1;
	pArgs->multipleOf		 = 256;
	pArgs->ffnDimMultiplier	 = -1.0f;
	pArgs->normEps			 = 1e-5f;
	pArgs->ropeTheta		 = 500000.0f;
	pArgs->useScaledRope	 = false;
	pArgs->maxSequenceLength = 2048;
	pArgs->nRep				 = 0;
	pArgs->headDim			 = pArgs->dim / pArgs->nHeads;

	return true;
}

bool lyGetModelTensor(lyTensor** ppTensor, const lyModel* pModel, const char* name)
{
	if (!ppTensor || !pModel || !name)
	{
		return false;
	}

	lyTensor* pModelTensor = NULL;
	for (int32_t i = 0; i < pModel->tensorCount; i++)
	{
		if (strcmp(pModel->tensors[i].name, name) == 0)
		{
			pModelTensor = &pModel->tensors[i];
			break;
		}
	}

	if (!pModelTensor)
	{
		return false;
	}

	lyTensor* pGpuTensor;
	if (!lyCreateTensor(&pGpuTensor, LY_MEMORY_CPU))
	{
		return false;
	}

	if (!lySetTensorShape(pGpuTensor, pModelTensor->shape, pModelTensor->rank))
	{
		lyDestroyTensor(pGpuTensor);
		return false;
	}

	if (!lySetTensorName(pGpuTensor, pModelTensor->name))
	{
		lyDestroyTensor(pGpuTensor);
		return false;
	}

	if (!lySetTensorData(pGpuTensor, pModelTensor->data, pModelTensor->dataSize))
	{
		lyDestroyTensor(pGpuTensor);
		return false;
	}

	*ppTensor = pGpuTensor;
	return true;
}