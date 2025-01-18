#include "lyTokenizer.h"
#include "lyTokenizerLoader.h"
#include "unity.h"

static lyTokenizer* pTokenizer = NULL;

void setUp(void)
{
	lyLoadTokenizer(&pTokenizer, "../model-tuned");
}

void tearDown(void)
{
	if (pTokenizer)
	{
		lyDestroyTokenizer(pTokenizer);
		pTokenizer = NULL;
	}
}

void test_SimpleTokenization(void)
{
	const char* text	   = "Hello world!";
	int32_t*	tokenIds   = NULL;
	int32_t		tokenCount = 0;

	TEST_ASSERT_TRUE(lyTokenizeText(&tokenIds, &tokenCount, pTokenizer, text, false));
	TEST_ASSERT_NOT_NULL(tokenIds);
	TEST_ASSERT_GREATER_THAN(0, tokenCount);

	char* reconstructed = NULL;
	TEST_ASSERT_TRUE(lyDetokenize(&reconstructed, pTokenizer, tokenIds, tokenCount));
	TEST_ASSERT_EQUAL_STRING("Hello world!", reconstructed);

	free(reconstructed);
	free(tokenIds);
}

void test_Contractions(void)
{
	const char* text	   = "I'm don't can't";
	int32_t*	tokenIds   = NULL;
	int32_t		tokenCount = 0;

	TEST_ASSERT_TRUE(lyTokenizeText(&tokenIds, &tokenCount, pTokenizer, text, false));

	char* reconstructed = NULL;
	TEST_ASSERT_TRUE(lyDetokenize(&reconstructed, pTokenizer, tokenIds, tokenCount));
	TEST_ASSERT_EQUAL_STRING(text, reconstructed);

	free(reconstructed);
	free(tokenIds);
}

void test_ChatPrompt(void)
{
	const char* systemPrompt = "You are a helpful assistant.";
	const char* userPrompt	 = "Tell me a story.";
	int32_t*	tokenIds	 = NULL;
	int32_t		tokenCount	 = 0;

	TEST_ASSERT_TRUE(lyTokenizePrompt(&tokenIds, &tokenCount, pTokenizer, systemPrompt, userPrompt));
	TEST_ASSERT_NOT_NULL(tokenIds);
	TEST_ASSERT_GREATER_THAN(0, tokenCount);

	bool foundBos	 = false;
	bool foundHeader = false;
	bool foundEot	 = false;

	for (int32_t i = 0; i < tokenCount; i++)
	{
		if (tokenIds[i] == pTokenizer->beginOfSentenceId)
			foundBos = true;
		if (tokenIds[i] == pTokenizer->endOfSentenceId)
			foundEot = true;
	}

	TEST_ASSERT_TRUE(foundBos);
	TEST_ASSERT_TRUE(foundEot);

	free(tokenIds);
}

void test_Numbers(void)
{
	const char* text	   = "123 4567 89";
	int32_t*	tokenIds   = NULL;
	int32_t		tokenCount = 0;

	TEST_ASSERT_TRUE(lyTokenizeText(&tokenIds, &tokenCount, pTokenizer, text, false));

	char* reconstructed = NULL;
	TEST_ASSERT_TRUE(lyDetokenize(&reconstructed, pTokenizer, tokenIds, tokenCount));

	TEST_ASSERT_EQUAL_STRING("123 4567 89", reconstructed);

	free(reconstructed);
	free(tokenIds);
}

void test_Whitespace(void)
{
	const char* text	   = "Hello\n\nWorld  !";
	int32_t*	tokenIds   = NULL;
	int32_t		tokenCount = 0;

	TEST_ASSERT_TRUE(lyTokenizeText(&tokenIds, &tokenCount, pTokenizer, text, false));

	char* reconstructed = NULL;
	TEST_ASSERT_TRUE(lyDetokenize(&reconstructed, pTokenizer, tokenIds, tokenCount));
	TEST_ASSERT_EQUAL_STRING(text, reconstructed);

	free(reconstructed);
	free(tokenIds);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_SimpleTokenization);
	RUN_TEST(test_Contractions);
	RUN_TEST(test_ChatPrompt);
	RUN_TEST(test_Numbers);
	RUN_TEST(test_Whitespace);
	return UNITY_END();
}