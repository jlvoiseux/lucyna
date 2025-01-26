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

void lyInferenceCreateInputTokens(lyTensor** ppTokens, const int32_t* tokenIds, int32_t tokenCount)
{
	int32_t	  shape[] = {tokenCount};
	lyTensor* tokens;
	lyTensorCreate(&tokens, shape, 1, NULL, NULL);

	for (int32_t i = 0; i < tokenCount; i++)
		lyTensorSetItem(tokens, &i, tokenIds[i]);

	*ppTokens = tokens;
}

void lyInferenceGenerateNextToken(lyGenerationStepResult* pResult, lyInference* pInference, lyTensor* pInputTokens, int32_t startPos)
{
	lyTensorPrint(pInputTokens);

	lyTensor* logits;
	lyTransformerForward(&logits, pInference->transformer, pInputTokens, startPos);
	lyTensorPrint(logits);

	lyTensor* lastLogits;
	lyTensorSlice(&lastLogits, logits, logits->shape[0] - 1, logits->shape[0]);
	lyTensorDestroy(logits);
	lyTensorPrint(lastLogits);

	lyTensor* maxToken;
	lyTensorArgmax(&maxToken, lastLogits, lastLogits->rank - 1);
	lyTensorDestroy(lastLogits);
	lyTensorPrint(maxToken);

	int32_t nextToken;
	int32_t loc[] = {0};
	lyTensorGetItem(&nextToken, maxToken, loc);
	lyTensorDestroy(maxToken);

	pResult->tokenId = nextToken;
	printf("Next token: %d\n", nextToken);

	if (nextToken == pInference->tokenizer->endOfSentenceId)
		pResult->state = GSFinishedByReachingEOS;
	else if (startPos + 1 >= pInference->sequenceLength)
		pResult->state = GSFinishedByReachingSeqLen;
	else
		pResult->state = GSInProgress;
}