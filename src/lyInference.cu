#include "lyInference.h"
#include "lyTensorMath.h"
#include "lyTokenizerLoader.h"

#include <stdio.h>
#include <stdlib.h>

bool lyCreateInference(lyInference** ppInference, lyModel* pModel, int32_t sequenceLength, bool (*logFn)(const char* format, ...), const char* modelDir)
{
	if (!ppInference || !pModel || sequenceLength <= 0 || !modelDir)
	{
		return false;
	}

	lyInference* pInference = (lyInference*)malloc(sizeof(lyInference));
	if (!pInference)
	{
		return false;
	}

	pInference->model		   = pModel;
	pInference->sequenceLength = sequenceLength;
	pInference->logFn		   = logFn;

	if (!lyCreateTransformer(&pInference->transformer, pModel))
	{
		free(pInference);
		return false;
	}

	if (!lyLoadTokenizer(&pInference->tokenizer, modelDir))
	{
		lyDestroyTransformer(pInference->transformer);
		free(pInference);
		return false;
	}

	*ppInference = pInference;
	return true;
}

void lyDestroyInference(lyInference* pInference)
{
	if (!pInference)
	{
		return;
	}

	if (pInference->transformer)
	{
		lyDestroyTransformer(pInference->transformer);
	}

	if (pInference->tokenizer)
	{
		lyDestroyTokenizer(pInference->tokenizer);
	}

	free(pInference);
}

bool lyCreateInferenceTokens(lyTensor** ppTokens, const lyInference* pInference, const int32_t* tokenIds, int32_t tokenCount)
{
	if (!ppTokens || !pInference || !tokenIds || tokenCount <= 0)
	{
		return false;
	}

	int32_t	  shape[] = {pInference->sequenceLength};
	lyTensor* tokens;
	lyCreateTensor(&tokens, shape, 1, NULL, NULL);

	for (int32_t i = 0; i < pInference->sequenceLength; i++)
	{
		if (!lyTensorSetItem(tokens, &i, -1))
		{
			lyDestroyTensor(tokens);
			return false;
		}
	}

	for (int32_t i = 0; i < tokenCount; i++)
	{
		if (!lyTensorSetItem(tokens, &i, tokenIds[i]))
		{
			lyDestroyTensor(tokens);
			return false;
		}
	}

	*ppTokens = tokens;
	return true;
}

bool lyGenerateNextToken(lyGenerationStepResult* pResult, lyInference* pInference, lyTensor* pInputTokens, int32_t startPos)
{
	if (!pResult || !pInference || !pInputTokens || startPos < 0)
	{
		return false;
	}

	lyTensor* logits;
	if (!lyTransformerForward(&logits, pInference->transformer, pInputTokens, startPos))
	{
		return false;
	}

	lyTensor* lastLogits;
	lyTensorSlice(&lastLogits, logits, logits->shape[0] - 1, logits->shape[0]);
	lyDestroyTensor(logits);

	lyTensor* maxToken;
	if (!lyTensorArgmax(&maxToken, lastLogits, lastLogits->rank - 1))
	{
		lyDestroyTensor(lastLogits);
		return false;
	}
	lyDestroyTensor(lastLogits);

	int32_t nextToken;
	if (!lyTensorGetItem(&nextToken, maxToken, 0))
	{
		lyDestroyTensor(maxToken);
		return false;
	}
	lyDestroyTensor(maxToken);

	pResult->tokenId = nextToken;

	if (nextToken == pInference->tokenizer->endOfSentenceId)
	{
		pResult->state = GSFinishedByReachingEOS;
	}
	else if (startPos + 1 >= pInference->sequenceLength)
	{
		pResult->state = GSFinishedByReachingSeqLen;
	}
	else
	{
		pResult->state = GSInProgress;
	}

	return true;
}