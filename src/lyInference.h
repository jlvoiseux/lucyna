#pragma once

#include "lyModel.h"
#include "lyTensor.h"
#include "lyTokenizer.h"
#include "lyTransformer.h"

typedef enum lyGenerationState
{
	GSInProgress			   = 1,
	GSFinishedByReachingEOS	   = 2,
	GSFinishedByReachingSeqLen = 3
} lyGenerationState;

typedef struct lyGenerationStepResult
{
	lyGenerationState state;
	int32_t			  tokenId;
} lyGenerationStepResult;

typedef struct lyInference
{
	lyModel*	   model;
	lyTransformer* transformer;
	lyTokenizer*   tokenizer;
	int32_t		   sequenceLength;
	bool (*logFn)(const char* format, ...);
} lyInference;

void lyInferenceCreate(lyInference** ppInference, lyModel* pModel, int32_t sequenceLength, bool (*logFn)(const char* format, ...), const char* modelDir);
void lyInferenceDestroy(lyInference* pInference);
void lyInferenceGenerateNextToken(lyGenerationStepResult* pResult, lyInference* pInference, lyTensor* pInputTokens, int32_t startPos);
void lyInferenceCreateInputTokens(lyTensor** ppTokens, const int32_t* tokenIds, int32_t tokenCount);