#pragma once

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