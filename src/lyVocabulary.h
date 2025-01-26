#pragma once

#include <stdint.h>
#include <stdlib.h>

typedef struct lyVocabEntry
{
	char*	token;
	int32_t rank;
} lyVocabEntry;

typedef struct lyVocabulary
{
	lyVocabEntry* entries;
	size_t		  count;
	size_t		  capacity;
} lyVocabulary;

void lyVocabularyCreate(lyVocabulary** ppVocab);
void lyVocabularyDestroy(lyVocabulary* pVocab);
void lyVocabularyAddEntry(lyVocabulary* pVocab, const char* token, int32_t rank);