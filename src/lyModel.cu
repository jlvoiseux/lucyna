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
	pArgs->headDim			 = 0;

	return true;
}