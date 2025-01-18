#pragma once

#include "lyTensor.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct lyToken
{
	char*	piece;
	int32_t rank;
} lyToken;

typedef struct lyTokenizer
{
	lyToken* tokens;
	int32_t	 tokenCount;

	int32_t	 beginOfSentenceId;
	int32_t	 endOfSentenceId;
	int32_t	 unknownId;
	int32_t	 padId;
	int32_t* stopTokenIds;
	int32_t	 stopTokenCount;
} lyTokenizer;

bool addToken(int32_t** ppTokenIds, int32_t* pTokenCount, int32_t tokenId);
bool findToken(int32_t* pTokenId, const lyTokenizer* pTokenizer, const char* piece);

bool lyTokenizeText(int32_t** ppTokenIds, int32_t* pTokenCount, const lyTokenizer* pTokenizer, const char* text, bool addBeginOfSentence);
bool lyTokenizePrompt(int32_t** ppTokenIds, int32_t* pTokenCount, const lyTokenizer* pTokenizer, const char* systemPrompt, const char* userPrompt);
bool lyDetokenize(char** ppText, const lyTokenizer* pTokenizer, const int32_t* tokenIds, int32_t tokenCount);