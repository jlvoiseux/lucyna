#include "lyTokenizer.h"
#include "lyUtil.h"
#include "lyVocabulary.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char* SPLIT_REGEX = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+";

static const char* SPECIAL_TOKENS[] = {"<|begin_of_text|>", "<|end_of_text|>", "<|reserved_special_token_0|>", "<|reserved_special_token_1|>", "<|finetune_right_pad_id|>", "<|step_id|>", "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>", "<|python_tag|>"};

static const char* B_TXT	= "<|begin_of_text|>";
static const char* B_HEADER = "<|start_header_id|>";
static const char* E_HEADER = "<|end_header_id|>";
static const char* E_TURN	= "<|eot_id|>";

#define NUM_SPECIAL_TOKENS (sizeof(SPECIAL_TOKENS) / sizeof(SPECIAL_TOKENS[0]))
#define RESERVED_SPECIAL_TOKENS_COUNT 256

typedef struct
{
	char*  bytes;
	size_t len;
} ByteSlice;

static void loadVocabulary(lyTokenizer* pTokenizer, const char* vocabPath)
{
	FILE* fp = fopen(vocabPath, "r");
	if (!fp)
	{
		fprintf(stderr, "Failed to open vocabulary file: %s\n", vocabPath);
		return;
	}

	lyVocabulary* pVocab;
	lyVocabularyCreate(&pVocab);

	char line[1024];
	while (fgets(line, sizeof(line), fp))
	{
		char* token	  = strtok(line, " ");
		char* rankStr = strtok(NULL, " \n");
		if (!token || !rankStr)
			continue;

		size_t		   outLen;
		unsigned char* decodedToken = lyBase64Decode(token, strlen(token), &outLen);
		int32_t		   rank			= atoi(rankStr);

		lyVocabularyAddEntry(pVocab, (const char*)decodedToken, rank);
		free(decodedToken);
	}

	pTokenizer->vocabSize = pVocab->count;
	pTokenizer->idToToken = (char**)malloc(sizeof(char*) * pVocab->count);
	pTokenizer->tokenToId = (int32_t*)malloc(sizeof(int32_t) * pVocab->count);

	for (size_t i = 0; i < pVocab->count; i++)
	{
		pTokenizer->idToToken[i] = strdup(pVocab->entries[i].token);
		pTokenizer->tokenToId[i] = pVocab->entries[i].rank;
	}

	lyVocabularyDestroy(pVocab);
	fclose(fp);
}

static void bytePairMerge(int32_t** ppTokens, size_t* pTokenCount, const char* piece, const lyTokenizer* pTokenizer)
{
	typedef struct
	{
		int32_t rank;
		size_t	idx;
	} RankTuple;

	size_t	   pieceLen = strlen(piece);
	RankTuple* parts	= (RankTuple*)malloc(sizeof(RankTuple) * (pieceLen + 1));
	RankTuple  minRank	= {INT32_MAX, SIZE_MAX};

	for (size_t i = 0; i < pieceLen - 1; i++)
	{
		char	pair[3] = {piece[i], piece[i + 1], '\0'};
		int32_t rank	= INT32_MAX;

		for (size_t j = 0; j < pTokenizer->vocabSize; j++)
		{
			if (strcmp(pTokenizer->idToToken[j], pair) == 0)
			{
				rank = pTokenizer->tokenToId[j];
				break;
			}
		}

		if (rank < minRank.rank)
		{
			minRank.rank = rank;
			minRank.idx	 = i;
		}
		parts[i].rank = rank;
		parts[i].idx  = i;
	}
	parts[pieceLen - 1].rank = INT32_MAX;
	parts[pieceLen - 1].idx	 = pieceLen - 1;
	parts[pieceLen].rank	 = INT32_MAX;
	parts[pieceLen].idx		 = pieceLen;

	size_t partCount = pieceLen + 1;
	while (minRank.rank != INT32_MAX && partCount > 1)
	{
		size_t i = minRank.idx;

		if (i > 0)
		{
			char merge[4];
			strncpy(merge, piece + parts[i - 1].idx, parts[i + 2].idx - parts[i - 1].idx);
			merge[parts[i + 2].idx - parts[i - 1].idx] = '\0';

			int32_t newRank = INT32_MAX;
			for (size_t j = 0; j < pTokenizer->vocabSize; j++)
			{
				if (strcmp(pTokenizer->idToToken[j], merge) == 0)
				{
					newRank = pTokenizer->tokenToId[j];
					break;
				}
			}
			parts[i - 1].rank = newRank;
		}

		memmove(&parts[i + 1], &parts[i + 2], sizeof(RankTuple) * (partCount - i - 2));
		partCount--;

		minRank.rank = INT32_MAX;
		minRank.idx	 = SIZE_MAX;
		for (size_t j = 0; j < partCount - 1; j++)
		{
			if (parts[j].rank < minRank.rank)
			{
				minRank.rank = parts[j].rank;
				minRank.idx	 = j;
			}
		}
	}

	*pTokenCount = partCount - 1;
	*ppTokens	 = (int32_t*)malloc(sizeof(int32_t) * *pTokenCount);

	for (size_t i = 0; i < *pTokenCount; i++)
	{
		char   subpiece[1024];
		size_t len = parts[i + 1].idx - parts[i].idx;
		strncpy(subpiece, piece + parts[i].idx, len);
		subpiece[len] = '\0';

		for (size_t j = 0; j < pTokenizer->vocabSize; j++)
		{
			if (strcmp(pTokenizer->idToToken[j], subpiece) == 0)
			{
				(*ppTokens)[i] = pTokenizer->tokenToId[j];
				break;
			}
		}
	}

	free(parts);
}

void lyTokenizerCreate(lyTokenizer** ppTokenizer, const char* modelFolderPath)
{
	lyTokenizer* pTokenizer = (lyTokenizer*)malloc(sizeof(lyTokenizer));
	memset(pTokenizer, 0, sizeof(lyTokenizer));

	// Compile regex
	int		   errorcode;
	PCRE2_SIZE erroroffset;
	pTokenizer->splitRegex = pcre2_compile((PCRE2_SPTR) "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+", PCRE2_ZERO_TERMINATED, PCRE2_UTF | PCRE2_UCP, &errorcode, &erroroffset, NULL);

	// Load base vocabulary
	char vocabPath[1024];
#ifdef _WIN32
	sprintf_s(vocabPath, sizeof(vocabPath), "%s\\tokenizer.model", modelFolderPath);
#else
	snprintf(vocabPath, sizeof(vocabPath), "%s/tokenizer.model", modelFolderPath);
#endif
	FILE* fp = fopen(vocabPath, "r");
	if (!fp)
	{
		fprintf(stderr, "Failed to open vocabulary file: %s\n", vocabPath);
		return;
	}

	// Count lines to determine vocabulary size
	size_t baseVocabSize = 0;
	char   line[1024];
	while (fgets(line, sizeof(line), fp))
	{
		baseVocabSize++;
	}
	rewind(fp);

	// Allocate vocabulary arrays
	size_t totalVocabSize = baseVocabSize + RESERVED_SPECIAL_TOKENS_COUNT;
	pTokenizer->idToToken = (char**)malloc(totalVocabSize * sizeof(char*));
	pTokenizer->tokenToId = (int32_t*)malloc(totalVocabSize * sizeof(int32_t));
	pTokenizer->vocabSize = totalVocabSize;

	// Load base vocabulary
	size_t idx = 0;
	while (fgets(line, sizeof(line), fp))
	{
		char* token	  = strtok(line, " ");
		char* rankStr = strtok(NULL, " \n");
		if (!token || !rankStr)
			continue;

		size_t		   outLen;
		unsigned char* decodedToken = lyBase64Decode(token, strlen(token), &outLen);
		int32_t		   rank			= atoi(rankStr);

		pTokenizer->idToToken[idx] = (char*)malloc(outLen + 1);
		memcpy(pTokenizer->idToToken[idx], decodedToken, outLen);
		pTokenizer->idToToken[idx][outLen] = '\0';
		pTokenizer->tokenToId[idx]		   = rank;

		free(decodedToken);
		idx++;
	}
	fclose(fp);

	size_t specialTokenBaseIdx = baseVocabSize;
	for (size_t i = 0; i < NUM_SPECIAL_TOKENS; i++)
	{
		pTokenizer->idToToken[specialTokenBaseIdx + i] = strdup(SPECIAL_TOKENS[i]);
		pTokenizer->tokenToId[specialTokenBaseIdx + i] = specialTokenBaseIdx + i;
	}

	// Add remaining reserved tokens
	for (size_t i = NUM_SPECIAL_TOKENS; i < RESERVED_SPECIAL_TOKENS_COUNT; i++)
	{
		char reserved[64];
		snprintf(reserved, sizeof(reserved), "<|reserved_special_token_%zu|>", i + 2);
		pTokenizer->idToToken[specialTokenBaseIdx + i] = strdup(reserved);
		pTokenizer->tokenToId[specialTokenBaseIdx + i] = specialTokenBaseIdx + i;
	}

	pTokenizer->beginOfSentenceId = specialTokenBaseIdx;	  // <|begin_of_text|>
	pTokenizer->endOfSentenceId	  = specialTokenBaseIdx + 1;  // <|end_of_text|>
	pTokenizer->padId			  = -1;
	pTokenizer->unknownId		  = -1;

	pTokenizer->stopTokenCount	= 2;
	pTokenizer->stopTokenIds	= (int32_t*)malloc(sizeof(int32_t) * pTokenizer->stopTokenCount);
	pTokenizer->stopTokenIds[0] = specialTokenBaseIdx + 8;	// <|eom_id|>
	pTokenizer->stopTokenIds[1] = specialTokenBaseIdx + 9;	// <|eot_id|>

	*ppTokenizer = pTokenizer;
}

void lyTokenizerDestroy(lyTokenizer* pTokenizer)
{
	if (!pTokenizer)
		return;

	pcre2_code_free(pTokenizer->splitRegex);

	for (size_t i = 0; i < pTokenizer->vocabSize; i++)
	{
		free(pTokenizer->idToToken[i]);
	}
	free(pTokenizer->idToToken);
	free(pTokenizer->tokenToId);
	free(pTokenizer->stopTokenIds);
	free(pTokenizer);
}

void lyTokenizerTokenize(int32_t** ppTokens, size_t* pTokenCount, const lyTokenizer* pTokenizer, const char* text, bool addBeginOfSentence)
{
	size_t maxTokens = strlen(text) * 2;  // Conservative estimate
	*ppTokens		 = (int32_t*)malloc(sizeof(int32_t) * maxTokens);
	*pTokenCount	 = 0;

	if (addBeginOfSentence && pTokenizer->beginOfSentenceId != -1)
	{
		(*ppTokens)[(*pTokenCount)++] = pTokenizer->beginOfSentenceId;
	}

	pcre2_match_data* match_data = pcre2_match_data_create_from_pattern(pTokenizer->splitRegex, NULL);

	int	   rc;
	size_t offset = 0;
	while ((rc = pcre2_match(pTokenizer->splitRegex, (PCRE2_SPTR)text, strlen(text), offset, 0, match_data, NULL)) > 0)
	{
		PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);

		size_t matchLen = ovector[1] - ovector[0];
		char*  match	= (char*)malloc(matchLen + 1);
		memcpy(match, text + ovector[0], matchLen);
		match[matchLen] = '\0';

		bool found = false;
		for (size_t i = 0; i < pTokenizer->vocabSize; i++)
		{
			if (strcmp(pTokenizer->idToToken[i], match) == 0)
			{
				(*ppTokens)[(*pTokenCount)++] = pTokenizer->tokenToId[i];
				found						  = true;
				break;
			}
		}

		if (!found)
		{
			int32_t* mergeTokens;
			size_t	 mergeCount;
			bytePairMerge(&mergeTokens, &mergeCount, match, pTokenizer);

			memcpy(*ppTokens + *pTokenCount, mergeTokens, mergeCount * sizeof(int32_t));
			*pTokenCount += mergeCount;

			free(mergeTokens);
		}

		free(match);
		offset = ovector[1];
	}

	pcre2_match_data_free(match_data);

	*ppTokens = (int32_t*)realloc(*ppTokens, sizeof(int32_t) * *pTokenCount);
}

void lyTokenizerTokenizePrompt(int32_t** ppTokens, size_t* pTokenCount, const lyTokenizer* pTokenizer, const char* systemPrompt, const char* userPrompt)
{
	size_t maxTokens = (systemPrompt ? strlen(systemPrompt) : 0) + (userPrompt ? strlen(userPrompt) : 0) * 2;
	*ppTokens		 = (int32_t*)malloc(sizeof(int32_t) * maxTokens);
	*pTokenCount	 = 0;

	// Add <|begin_of_text|>
	(*ppTokens)[(*pTokenCount)++] = pTokenizer->beginOfSentenceId;

	// Add system message if present
	if (systemPrompt)
	{
		// Add system header
		for (size_t i = 0; i < pTokenizer->vocabSize; i++)
		{
			if (strcmp(pTokenizer->idToToken[i], "<|start_header_id|>") == 0)
			{
				(*ppTokens)[(*pTokenCount)++] = pTokenizer->tokenToId[i];
				break;
			}
		}

		int32_t* tokens;
		size_t	 tokenCount;
		lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, "system", false);
		memcpy(*ppTokens + *pTokenCount, tokens, tokenCount * sizeof(int32_t));
		*pTokenCount += tokenCount;
		free(tokens);

		for (size_t i = 0; i < pTokenizer->vocabSize; i++)
		{
			if (strcmp(pTokenizer->idToToken[i], "<|end_header_id|>") == 0)
			{
				(*ppTokens)[(*pTokenCount)++] = pTokenizer->tokenToId[i];
				break;
			}
		}

		// Add newlines
		lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, "\n\n", false);
		memcpy(*ppTokens + *pTokenCount, tokens, tokenCount * sizeof(int32_t));
		*pTokenCount += tokenCount;
		free(tokens);

		// Add system content
		lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, systemPrompt, false);
		memcpy(*ppTokens + *pTokenCount, tokens, tokenCount * sizeof(int32_t));
		*pTokenCount += tokenCount;
		free(tokens);

		// Add end of turn
		for (size_t i = 0; i < pTokenizer->vocabSize; i++)
		{
			if (strcmp(pTokenizer->idToToken[i], "<|eot_id|>") == 0)
			{
				(*ppTokens)[(*pTokenCount)++] = pTokenizer->tokenToId[i];
				break;
			}
		}
	}

	// Add user message if present
	if (userPrompt)
	{
		// Add user header
		for (size_t i = 0; i < pTokenizer->vocabSize; i++)
		{
			if (strcmp(pTokenizer->idToToken[i], "<|start_header_id|>") == 0)
			{
				(*ppTokens)[(*pTokenCount)++] = pTokenizer->tokenToId[i];
				break;
			}
		}

		int32_t* tokens;
		size_t	 tokenCount;
		lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, "user", false);
		memcpy(*ppTokens + *pTokenCount, tokens, tokenCount * sizeof(int32_t));
		*pTokenCount += tokenCount;
		free(tokens);

		for (size_t i = 0; i < pTokenizer->vocabSize; i++)
		{
			if (strcmp(pTokenizer->idToToken[i], "<|end_header_id|>") == 0)
			{
				(*ppTokens)[(*pTokenCount)++] = pTokenizer->tokenToId[i];
				break;
			}
		}

		lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, "\n\n", false);
		memcpy(*ppTokens + *pTokenCount, tokens, tokenCount * sizeof(int32_t));
		*pTokenCount += tokenCount;
		free(tokens);

		// Add user content
		lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, userPrompt, false);
		memcpy(*ppTokens + *pTokenCount, tokens, tokenCount * sizeof(int32_t));
		*pTokenCount += tokenCount;
		free(tokens);

		// Add end of turn
		for (size_t i = 0; i < pTokenizer->vocabSize; i++)
		{
			if (strcmp(pTokenizer->idToToken[i], "<|eot_id|>") == 0)
			{
				(*ppTokens)[(*pTokenCount)++] = pTokenizer->tokenToId[i];
				break;
			}
		}
	}

	// Add assistant header
	for (size_t i = 0; i < pTokenizer->vocabSize; i++)
	{
		if (strcmp(pTokenizer->idToToken[i], "<|start_header_id|>") == 0)
		{
			(*ppTokens)[(*pTokenCount)++] = pTokenizer->tokenToId[i];
			break;
		}
	}

	int32_t* tokens;
	size_t	 tokenCount;
	lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, "assistant", false);
	memcpy(*ppTokens + *pTokenCount, tokens, tokenCount * sizeof(int32_t));
	*pTokenCount += tokenCount;
	free(tokens);

	for (size_t i = 0; i < pTokenizer->vocabSize; i++)
	{
		if (strcmp(pTokenizer->idToToken[i], "<|end_header_id|>") == 0)
		{
			(*ppTokens)[(*pTokenCount)++] = pTokenizer->tokenToId[i];
			break;
		}
	}

	lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, "\n\n", false);
	memcpy(*ppTokens + *pTokenCount, tokens, tokenCount * sizeof(int32_t));
	*pTokenCount += tokenCount;
	free(tokens);

	*ppTokens = (int32_t*)realloc(*ppTokens, sizeof(int32_t) * *pTokenCount);
}

void lyTokenizerDecodeToken(char** ppStr, const lyTokenizer* pTokenizer, int32_t tokenId)
{
	if (tokenId < 0 || tokenId >= (int32_t)pTokenizer->vocabSize)
	{
		*ppStr = strdup("<UNKNOWN>");
		return;
	}

	*ppStr = strdup(pTokenizer->idToToken[tokenId]);
}

void lyTokenizerDecodeBatch(char** ppStr, const lyTokenizer* pTokenizer, const int32_t* tokens, size_t tokenCount)
{
	size_t totalLen = 0;
	for (size_t i = 0; i < tokenCount; i++)
	{
		if (tokens[i] == pTokenizer->padId)
			break;
		char* token;
		lyTokenizerDecodeToken(&token, pTokenizer, tokens[i]);
		totalLen += strlen(token);
		free(token);
	}

	*ppStr		= (char*)malloc(totalLen + 1);
	(*ppStr)[0] = '\0';

	for (size_t i = 0; i < tokenCount; i++)
	{
		if (tokens[i] == pTokenizer->padId)
			break;
		char* token;
		lyTokenizerDecodeToken(&token, pTokenizer, tokens[i]);
		strcat(*ppStr, token);
		free(token);
	}
}