#include "lyModel.h"

#include <stdlib.h>
#include <string.h>

void lyModelCreateDefaultArgs(lyModelArgs* pArgs)
{
	pArgs->dim				 = 4096;
	pArgs->nLayers			 = 32;
	pArgs->nHeads			 = 32;
	pArgs->nKVHeads			 = -1;
	pArgs->vocabSize		 = -1;
	pArgs->multipleOf		 = 256;
	pArgs->ffnDimMultiplier	 = -1.0f;
	pArgs->normEps			 = 1e-5f;
	pArgs->ropeTheta		 = 500000.0f;
	pArgs->maxSequenceLength = 2048;
	pArgs->nRep				 = 0;
	pArgs->headDim			 = pArgs->dim / pArgs->nHeads;
	pArgs->useScaledRope	 = false;
}

void lyModelGetTensor(lyTensor** ppTensor, const lyModel* pModel, const char* name)
{
	lyTensor* pModelTensor = NULL;
	for (int32_t i = 0; i < pModel->tensorCount; i++)
	{
		if (strcmp(pModel->tensors[i]->name, name) == 0)
		{
			pModelTensor = pModel->tensors[i];
			break;
		}
	}

	*ppTensor = pModelTensor;
}