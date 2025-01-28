#include "lyTokenizer.h"
#include "lyUtil.h"
#include "unity.h"

static lyTokenizer* pTokenizer = NULL;

void setUp(void)
{
	lyTokenizerCreate(&pTokenizer, "../model-tuned");
}

void tearDown(void)
{
	if (pTokenizer)
	{
		lyTokenizerDestroy(pTokenizer);
		pTokenizer = NULL;
	}
}

void test_TokenizerBasic(void)
{
	const char* text = "Hello world";
	int32_t*	tokens;
	size_t		tokenCount;

	lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, text, false);

	TEST_ASSERT_NOT_NULL(tokens);
	TEST_ASSERT_GREATER_THAN(0, tokenCount);

	char* decoded;
	lyTokenizerDecodeBatch(&decoded, pTokenizer, tokens, tokenCount);
	TEST_ASSERT_EQUAL_STRING("Hello world", decoded);

	free(tokens);
	free(decoded);
}

void test_TokenizerBase64(void)
{
	size_t		   outLen;
	unsigned char* decoded = lyBase64Decode("SGVsbG8=", 8, &outLen);
	TEST_ASSERT_EQUAL_STRING("Hello", (char*)decoded);
	free(decoded);
}

void test_TokenizerBytePairMerge(void)
{
	const char* text = "Hello";
	int32_t*	tokens;
	size_t		tokenCount;

	lyTokenizerTokenize(&tokens, &tokenCount, pTokenizer, text, false);

	// The exact token IDs will depend on the vocabulary,
	// but we can check that we got some tokens
	TEST_ASSERT_GREATER_THAN(0, tokenCount);

	char* decoded;
	lyTokenizerDecodeBatch(&decoded, pTokenizer, tokens, tokenCount);
	TEST_ASSERT_EQUAL_STRING("Hello", decoded);

	free(tokens);
	free(decoded);
}

void test_TokenizerPromptTemplate(void)
{
	const char* systemPrompt = "You are Einstein";
	const char* userPrompt	 = "Describe your theory.";

	int32_t* tokens;
	int32_t	 tokenCount;

	lyTokenizerTokenizePrompt(&tokens, &tokenCount, pTokenizer, systemPrompt, userPrompt);

	char* decodedStr;
	lyTokenizerDecodeBatch(&decodedStr, pTokenizer, tokens, tokenCount);

	const char* expected = "<|begin_of_text|>"
						   "<|start_header_id|>system<|end_header_id|>\n\n"
						   "You are Einstein<|eot_id|>"
						   "<|start_header_id|>user<|end_header_id|>\n\n"
						   "Describe your theory.<|eot_id|>"
						   "<|start_header_id|>assistant<|end_header_id|>\n\n";

	TEST_ASSERT_EQUAL_STRING(expected, decodedStr);
	free(tokens);
	free(decodedStr);
}

int main(void)
{
	UNITY_BEGIN();
	RUN_TEST(test_TokenizerBasic);
	RUN_TEST(test_TokenizerBase64);
	RUN_TEST(test_TokenizerBytePairMerge);
	RUN_TEST(test_TokenizerPromptTemplate);
	return UNITY_END();
}