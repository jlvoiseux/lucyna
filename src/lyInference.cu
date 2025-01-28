#include "lyInference.h"
#include "lyTensorMath.h"

#include <stdio.h>
#include <stdlib.h>

void lyInferenceCreate(lyInference** ppInference, lyModel* pModel, int32_t sequenceLength, bool (*logFn)(const char* format, ...), const char* modelDir)
{
	lyInference* pInference = (lyInference*)malloc(sizeof(lyInference));

	pInference->model		   = pModel;
	pInference->sequenceLength = sequenceLength;
	pInference->logFn		   = logFn;

	lyTransformerCreate(&pInference->transformer, pModel);
	lyTokenizerCreate(&pInference->tokenizer, "../model-tuned");

	*ppInference = pInference;
}

void lyInferenceDestroy(lyInference* pInference)
{
	if (pInference->transformer)
		lyTransformerDestroy(pInference->transformer);

	if (pInference->tokenizer)
		lyTokenizerDestroy(pInference->tokenizer);

	free(pInference);
}

void lyInferenceGenerateNextToken(lyGenerationStepResult* pResult, lyInference* pInference, int32_t* pTokens, int32_t* pTokenCount, int32_t startPos)
{
	if (*pTokenCount >= pInference->sequenceLength)
	{
		pResult->state = GSFinishedByReachingSeqLen;
		return;
	}

	int32_t	  curPos = *pTokenCount;
	lyTensor* logits;
	lyTransformerForward(&logits, pInference->transformer, pTokens + startPos, curPos - startPos, startPos);

	lyTensor* lastLogits;
	lyTensorSlice(&lastLogits, logits, logits->shape[0] - 1, logits->shape[0]);
	lyTensorDestroy(logits);

	int32_t nextToken;
	lyTensorArgmax(&nextToken, lastLogits);
	lyTensorDestroy(lastLogits);

	if (pTokens[curPos] != pInference->tokenizer->padId)
	{
		nextToken = pTokens[curPos];
	}

	pTokens[curPos] = nextToken;
	(*pTokenCount)++;

	pResult->tokenId = nextToken;

	bool isStopToken = false;
	for (size_t i = 0; i < pInference->tokenizer->stopTokenCount; i++)
	{
		if (nextToken == pInference->tokenizer->stopTokenIds[i])
		{
			isStopToken = true;
			break;
		}
	}

	if (isStopToken)
		pResult->state = GSFinishedByReachingEOS;
	else if (curPos + 1 >= pInference->sequenceLength)
		pResult->state = GSFinishedByReachingSeqLen;
	else
		pResult->state = GSInProgress;
}