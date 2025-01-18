#include "lyTokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* CONTRACTIONS[] = {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
#define NUM_CONTRACTIONS (sizeof(CONTRACTIONS) / sizeof(char*))

bool addToken(int32_t** ppTokenIds, int32_t* pTokenCount, int32_t tokenId)
{
	int32_t* newTokens = (int32_t*)realloc(*ppTokenIds, (*pTokenCount + 1) * sizeof(int32_t));
	if (!newTokens)
	{
		return false;
	}
	*ppTokenIds					= newTokens;
	(*ppTokenIds)[*pTokenCount] = tokenId;
	(*pTokenCount)++;
	return true;
}

bool findToken(int32_t* pTokenId, const lyTokenizer* pTokenizer, const char* piece)
{
	for (int32_t i = 0; i < pTokenizer->tokenCount; i++)
	{
		if (strcmp(pTokenizer->tokens[i].piece, piece) == 0)
		{
			*pTokenId = pTokenizer->tokens[i].rank;
			return true;
		}
	}
	return false;
}

static bool isContraction(const char* text)
{
	for (size_t i = 0; i < NUM_CONTRACTIONS; i++)
	{
		if (strncmp(text, CONTRACTIONS[i], strlen(CONTRACTIONS[i])) == 0)
		{
			return true;
		}
	}
	return false;
}

static bool isLetter(char c)
{
	return isalpha(c);
}

static bool isNumber(char c)
{
	return isdigit(c);
}

static size_t getNextToken(const char* text, char* token, size_t maxTokenLen)
{
	size_t pos		= 0;
	size_t tokenLen = 0;

	if (text[pos] == '\'' && isContraction(text + pos))
	{
		const char* space		   = strchr(text + pos, ' ');
		size_t		contractionLen = space ? (size_t)(space - (text + pos)) : strlen(text + pos);
		strncpy(token, text + pos, contractionLen);
		token[contractionLen] = '\0';
		return contractionLen;
	}

	if (isNumber(text[pos]))
	{
		while (pos < 3 && text[pos] && isNumber(text[pos]))
		{
			token[tokenLen++] = text[pos++];
		}
		token[tokenLen] = '\0';
		return pos;
	}

	if (isLetter(text[pos]))
	{
		if (pos == 0 && text[pos + 1] && isLetter(text[pos + 1]) && !isLetter(text[pos]) && !isNumber(text[pos]))
		{
			token[tokenLen++] = text[pos++];
		}

		while (text[pos] && isLetter(text[pos]))
		{
			token[tokenLen++] = text[pos++];
		}
		token[tokenLen] = '\0';
		return pos;
	}

	if (isspace(text[pos]))
	{
		if (text[pos] == '\n' || text[pos] == '\r')
		{
			while (text[pos] && (text[pos] == '\n' || text[pos] == '\r'))
			{
				token[tokenLen++] = text[pos++];
			}
		}
		else
		{
			while (text[pos] && isspace(text[pos]) && text[pos] != '\n' && text[pos] != '\r')
			{
				token[tokenLen++] = text[pos++];
			}
		}
		token[tokenLen] = '\0';
		return pos;
	}

	if (text[pos])
	{
		if (isspace(text[pos]))
		{
			token[tokenLen++] = text[pos++];
		}

		while (text[pos] && !isspace(text[pos]) && !isLetter(text[pos]) && !isNumber(text[pos]))
		{
			token[tokenLen++] = text[pos++];
		}

		while (text[pos] && (text[pos] == '\n' || text[pos] == '\r'))
		{
			token[tokenLen++] = text[pos++];
		}

		token[tokenLen] = '\0';
		return pos;
	}

	return 0;
}

bool lyTokenizeText(int32_t** ppTokenIds, int32_t* pTokenCount, const lyTokenizer* pTokenizer, const char* text, bool addBeginOfSentence)
{
	if (!ppTokenIds || !pTokenCount || !pTokenizer || !text)
	{
		return false;
	}

	*ppTokenIds	 = NULL;
	*pTokenCount = 0;

	if (addBeginOfSentence && pTokenizer->beginOfSentenceId >= 0)
	{
		if (!addToken(ppTokenIds, pTokenCount, pTokenizer->beginOfSentenceId))
		{
			return false;
		}
	}

	const char* p = text;
	char		token[1024];

	while (*p)
	{
		size_t advance = getNextToken(p, token, sizeof(token));
		if (advance == 0)
			break;

		int32_t tokenId;
		if (findToken(&tokenId, pTokenizer, token))
		{
			if (!addToken(ppTokenIds, pTokenCount, tokenId))
			{
				free(*ppTokenIds);
				return false;
			}
		}
		else if (pTokenizer->unknownId >= 0)
		{
			if (!addToken(ppTokenIds, pTokenCount, pTokenizer->unknownId))
			{
				free(*ppTokenIds);
				return false;
			}
		}

		p += advance;
	}

	return true;
}

bool lyTokenizePrompt(int32_t** ppTokenIds, int32_t* pTokenCount, const lyTokenizer* pTokenizer, const char* systemPrompt, const char* userPrompt)
{
	if (!ppTokenIds || !pTokenCount || !pTokenizer || !userPrompt)
	{
		return false;
	}

	*ppTokenIds	 = NULL;
	*pTokenCount = 0;

	// Format: <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{systemPrompt}<|eot_id|><|end_of_text|>
	//         <|start_header_id|>user<|end_header_id|>\n\n{userPrompt}<|eot_id|><|end_of_text|>
	//         <|start_header_id|>assistant<|end_header_id|>\n\n<|end_of_text|>

	if (!addToken(ppTokenIds, pTokenCount, pTokenizer->beginOfSentenceId))
	{
		return false;
	}

	int32_t startHeaderId, endHeaderId, eotId;
	if (!findToken(&startHeaderId, pTokenizer, "<|start_header_id|>") || !findToken(&endHeaderId, pTokenizer, "<|end_header_id|>") || !findToken(&eotId, pTokenizer, "<|eot_id|>"))
	{
		free(*ppTokenIds);
		return false;
	}

	if (systemPrompt && *systemPrompt)
	{
		if (!addToken(ppTokenIds, pTokenCount, startHeaderId) || !addToken(ppTokenIds, pTokenCount, endHeaderId))
		{
			free(*ppTokenIds);
			return false;
		}

		int32_t* systemTokens	  = NULL;
		int32_t	 systemTokenCount = 0;
		if (!lyTokenizeText(&systemTokens, &systemTokenCount, pTokenizer, systemPrompt, false))
		{
			free(*ppTokenIds);
			return false;
		}

		for (int32_t i = 0; i < systemTokenCount; i++)
		{
			if (!addToken(ppTokenIds, pTokenCount, systemTokens[i]))
			{
				free(systemTokens);
				free(*ppTokenIds);
				return false;
			}
		}
		free(systemTokens);

		if (!addToken(ppTokenIds, pTokenCount, eotId) || !addToken(ppTokenIds, pTokenCount, pTokenizer->endOfSentenceId))
		{
			free(*ppTokenIds);
			return false;
		}
	}

	if (!addToken(ppTokenIds, pTokenCount, startHeaderId) || !addToken(ppTokenIds, pTokenCount, endHeaderId))
	{
		free(*ppTokenIds);
		return false;
	}

	int32_t* userTokens		= NULL;
	int32_t	 userTokenCount = 0;
	if (!lyTokenizeText(&userTokens, &userTokenCount, pTokenizer, userPrompt, false))
	{
		free(*ppTokenIds);
		return false;
	}

	for (int32_t i = 0; i < userTokenCount; i++)
	{
		if (!addToken(ppTokenIds, pTokenCount, userTokens[i]))
		{
			free(userTokens);
			free(*ppTokenIds);
			return false;
		}
	}
	free(userTokens);

	if (!addToken(ppTokenIds, pTokenCount, eotId) || !addToken(ppTokenIds, pTokenCount, pTokenizer->endOfSentenceId) || !addToken(ppTokenIds, pTokenCount, startHeaderId) || !addToken(ppTokenIds, pTokenCount, endHeaderId) || !addToken(ppTokenIds, pTokenCount, pTokenizer->endOfSentenceId))
	{
		free(*ppTokenIds);
		return false;
	}

	return true;
}

bool lyDetokenize(char** ppText, const lyTokenizer* pTokenizer, const int32_t* tokenIds, int32_t tokenCount)
{
	if (!ppText || !pTokenizer || !tokenIds || tokenCount <= 0)
	{
		return false;
	}

	size_t totalLen = 1;
	for (int32_t i = 0; i < tokenCount; i++)
	{
		for (int32_t j = 0; j < pTokenizer->tokenCount; j++)
		{
			if (pTokenizer->tokens[j].rank == tokenIds[i])
			{
				totalLen += strlen(pTokenizer->tokens[j].piece);
				break;
			}
		}
	}

	char* text = (char*)malloc(totalLen);
	if (!text)
	{
		return false;
	}

	char* p = text;
	for (int32_t i = 0; i < tokenCount; i++)
	{
		for (int32_t j = 0; j < pTokenizer->tokenCount; j++)
		{
			if (pTokenizer->tokens[j].rank == tokenIds[i])
			{
				const char* piece	 = pTokenizer->tokens[j].piece;
				size_t		pieceLen = strlen(piece);
				memcpy(p, piece, pieceLen);
				p += pieceLen;
				break;
			}
		}
	}
	*p = '\0';

	*ppText = text;
	return true;
}