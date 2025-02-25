#include "lyVocabulary.h"

#include <string.h>

void lyVocabularyCreate(lyVocabulary** ppVocab)
{
	lyVocabulary* pVocab = (lyVocabulary*)malloc(sizeof(lyVocabulary));
	pVocab->capacity	 = 1024;
	pVocab->count		 = 0;
	pVocab->entries		 = (lyVocabEntry*)malloc(sizeof(lyVocabEntry) * pVocab->capacity);
	*ppVocab			 = pVocab;
}

void lyVocabularyDestroy(lyVocabulary* pVocab)
{
	if (!pVocab)
		return;

	for (size_t i = 0; i < pVocab->count; i++)
	{
		free(pVocab->entries[i].token);
	}
	free(pVocab->entries);
	free(pVocab);
}

void lyVocabularyAddEntry(lyVocabulary* pVocab, const char* token, int32_t rank)
{
	if (pVocab->count >= pVocab->capacity)
	{
		pVocab->capacity *= 2;
		pVocab->entries = (lyVocabEntry*)realloc(pVocab->entries, sizeof(lyVocabEntry) * pVocab->capacity);
	}

	pVocab->entries[pVocab->count].token = strdup(token);
	pVocab->entries[pVocab->count].rank	 = rank;
	pVocab->count++;
}