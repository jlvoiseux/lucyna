#pragma once

#include "lyTensor.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <stdbool.h>

typedef struct lyTokenizer
{
	pcre2_code* splitRegex;
	char**		idToToken;
	int32_t*	tokenToId;
	size_t		vocabSize;

	int32_t	 beginOfSentenceId;
	int32_t	 endOfSentenceId;
	int32_t	 unknownId;
	int32_t	 padId;
	int32_t* stopTokenIds;
	size_t	 stopTokenCount;
} lyTokenizer;

void lyTokenizerCreate(lyTokenizer** ppTokenizer, const char* modelFolderPath);
void lyTokenizerDestroy(lyTokenizer* pTokenizer);

void lyTokenizerTokenize(int32_t** ppTokens, size_t* pTokenCount, const lyTokenizer* pTokenizer, const char* text, bool addBeginOfSentence);
void lyTokenizerTokenizePrompt(int32_t** ppTokens, int32_t* pTokenCount, const lyTokenizer* pTokenizer, const char* systemPrompt, const char* userPrompt);

void lyTokenizerDecodeToken(char** ppStr, const lyTokenizer* pTokenizer, int32_t tokenId);
void lyTokenizerDecodeBatch(char** ppStr, const lyTokenizer* pTokenizer, const int32_t* tokens, int32_t tokenCount);