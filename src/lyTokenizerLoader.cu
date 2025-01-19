#include "lyTokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BASE64_DECODE_ERROR (-1)

static const char* SPECIAL_TOKENS[] = {"<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>", "<|reserved_special_token_1|>", "<|finetune_right_pad_id|>", "<|step_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>", "<|python_tag|>"};

#define SPECIAL_TOKENS_COUNT (sizeof(SPECIAL_TOKENS) / sizeof(char*))
#define RESERVED_SPECIAL_TOKENS_COUNT 256

static int base64Decode(const char* input, uint8_t* output)
{
	static const int lookup[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1,
								 -1, 0,	 1,	 2,	 3,	 4,	 5,	 6,	 7,	 8,	 9,	 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1, -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1};

	int len = strlen(input);
	if (len % 4 != 0)
		return BASE64_DECODE_ERROR;

	int outputLen = len / 4 * 3;
	if (input[len - 1] == '=')
		outputLen--;
	if (input[len - 2] == '=')
		outputLen--;

	for (int i = 0, j = 0; i < len;)
	{
		uint32_t a = input[i] == '=' ? 0 & i++ : lookup[input[i++]];
		uint32_t b = input[i] == '=' ? 0 & i++ : lookup[input[i++]];
		uint32_t c = input[i] == '=' ? 0 & i++ : lookup[input[i++]];
		uint32_t d = input[i] == '=' ? 0 & i++ : lookup[input[i++]];

		uint32_t triple = (a << 18) + (b << 12) + (c << 6) + d;

		if (j < outputLen)
			output[j++] = (triple >> 16) & 0xFF;
		if (j < outputLen)
			output[j++] = (triple >> 8) & 0xFF;
		if (j < outputLen)
			output[j++] = triple & 0xFF;
	}

	return outputLen;
}

void lyDestroyTokenizer(lyTokenizer* pTokenizer)
{
	if (!pTokenizer)
		return;

	if (pTokenizer->tokens)
	{
		for (int32_t i = 0; i < pTokenizer->tokenCount; i++)
		{
			free(pTokenizer->tokens[i].piece);
		}
		free(pTokenizer->tokens);
	}

	free(pTokenizer->stopTokenIds);
	free(pTokenizer);
}

bool lyLoadTokenizer(lyTokenizer** ppTokenizer, const char* modelDir)
{
	if (!ppTokenizer || !modelDir)
		return false;

	char tokenPath[1024];
#ifdef _WIN32
	sprintf_s(tokenPath, sizeof(tokenPath), "%s\\tokenizer.model", modelDir);
#else
	snprintf(tokenPath, sizeof(tokenPath), "%s/tokenizer.model", modelDir);
#endif

	FILE* file = fopen(tokenPath, "r");
	if (!file)
		return false;

	lyTokenizer* pTokenizer = (lyTokenizer*)malloc(sizeof(lyTokenizer));
	if (!pTokenizer)
	{
		fclose(file);
		return false;
	}
	memset(pTokenizer, 0, sizeof(lyTokenizer));

	int	 lineCount = 0;
	char ch;
	while (!feof(file))
	{
		ch = fgetc(file);
		if (ch == '\n')
			lineCount++;
	}
	rewind(file);

	pTokenizer->tokenCount = lineCount + RESERVED_SPECIAL_TOKENS_COUNT;
	pTokenizer->tokens	   = (lyToken*)malloc(sizeof(lyToken) * pTokenizer->tokenCount);
	if (!pTokenizer->tokens)
	{
		lyDestroyTokenizer(pTokenizer);
		fclose(file);
		return false;
	}

	char	line[1024];
	int		tokenIdx = 0;
	uint8_t decodedData[1024];

	while (fgets(line, sizeof(line), file))
	{
		char* space = strchr(line, ' ');
		if (!space)
			continue;

		*space	 = '\0';
		int rank = atoi(space + 1);

		int decodedLen = base64Decode(line, decodedData);
		if (decodedLen == BASE64_DECODE_ERROR)
			continue;

		decodedData[decodedLen] = '\0';

		pTokenizer->tokens[tokenIdx].piece = (char*)malloc(decodedLen + 1);
		if (!pTokenizer->tokens[tokenIdx].piece)
		{
			lyDestroyTokenizer(pTokenizer);
			fclose(file);
			return false;
		}

		memcpy(pTokenizer->tokens[tokenIdx].piece, decodedData, decodedLen + 1);
		pTokenizer->tokens[tokenIdx].rank = rank;
		tokenIdx++;
	}

	fclose(file);

	int baseTokenCount = tokenIdx;

	for (size_t i = 0; i < SPECIAL_TOKENS_COUNT; i++)
	{
		pTokenizer->tokens[tokenIdx].piece = strdup(SPECIAL_TOKENS[i]);
		pTokenizer->tokens[tokenIdx].rank  = baseTokenCount + i;
		tokenIdx++;
	}

	for (int i = 0; i < RESERVED_SPECIAL_TOKENS_COUNT - SPECIAL_TOKENS_COUNT; i++)
	{
		char reserved[64];
		sprintf(reserved, "<|reserved_special_token_%d|>", i + 2);
		pTokenizer->tokens[tokenIdx].piece = strdup(reserved);
		pTokenizer->tokens[tokenIdx].rank  = baseTokenCount + SPECIAL_TOKENS_COUNT + i;
		tokenIdx++;
	}

	for (tokenIdx = 0; tokenIdx < pTokenizer->tokenCount; tokenIdx++)
	{
		if (strcmp(pTokenizer->tokens[tokenIdx].piece, "<|begin_of_text|>") == 0)
			pTokenizer->beginOfSentenceId = pTokenizer->tokens[tokenIdx].rank;
		else if (strcmp(pTokenizer->tokens[tokenIdx].piece, "<|end_of_text|>") == 0)
			pTokenizer->endOfSentenceId = pTokenizer->tokens[tokenIdx].rank;
	}

	pTokenizer->padId	  = -1;
	pTokenizer->unknownId = -1;

	pTokenizer->stopTokenCount = 2;
	pTokenizer->stopTokenIds   = (int32_t*)malloc(sizeof(int32_t) * pTokenizer->stopTokenCount);
	if (!pTokenizer->stopTokenIds)
	{
		lyDestroyTokenizer(pTokenizer);
		return false;
	}

	int stopTokenIdx = 0;
	for (tokenIdx = 0; tokenIdx < pTokenizer->tokenCount; tokenIdx++)
	{
		if (strcmp(pTokenizer->tokens[tokenIdx].piece, "<|eom_id|>") == 0 || strcmp(pTokenizer->tokens[tokenIdx].piece, "<|eot_id|>") == 0)
		{
			pTokenizer->stopTokenIds[stopTokenIdx++] = pTokenizer->tokens[tokenIdx].rank;
		}
	}

	printf("\nTokenizer loaded with %d tokens\n", pTokenizer->tokenCount);
	
	*ppTokenizer = pTokenizer;
	return true;
}