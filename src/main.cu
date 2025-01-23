#include "lyAttention.h"
#include "lyInference.h"
#include "lyModelLoader.h"
#include "lyTokenizerLoader.h"
#include "lyUtil.h"

#include <stdio.h>

static bool getInput(char* buffer)
{
	printf("%s", "Enter your message (or press enter to quit): ");
	if (!fgets(buffer, 2048, stdin))
	{
		return false;
	}

	size_t len = strlen(buffer);
	if (len > 0 && buffer[len - 1] == '\n')
	{
		buffer[len - 1] = '\0';
	}

	return true;
}

static bool logCallback(const char* format, ...)
{
	va_list args;
	va_start(args, format);
	vprintf(format, args);
	va_end(args);
	printf("\n");
	return true;
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Usage: %s <model_dir>\n", argv[0]);
		printf("Example: %s models/Llama-3.2b-chat\n", argv[0]);
		return 1;
	}

	printDeviceInfo();

	lyModel* pModel;
	if (!lyLoadModel(&pModel, argv[1], true, true))
	{
		printf("Failed to load model from directory: %s\n", argv[1]);
		return 1;
	}

	printf("Model loaded successfully.\n\n");

	lyInference* pInference;
	if (!lyCreateInference(&pInference, pModel, 2048, logCallback, argv[1]))
	{
		printf("Failed to create inference engine.\n");
		lyDestroyModel(pModel);
		return 1;
	}

	char prompt[2048];
	while (getInput(prompt))
	{
		if (strlen(prompt) == 0)
		{
			break;
		}

		int32_t* tokenIds	= NULL;
		int32_t	 tokenCount = 0;
		if (!lyTokenizePrompt(&tokenIds, &tokenCount, pInference->tokenizer, NULL, prompt))
		{
			printf("Failed to tokenize prompt.\n");
			continue;
		}

		lyTensor* pInputTokens;
		if (!lyCreateInferenceTokens(&pInputTokens, pInference, tokenIds, tokenCount))
		{
			printf("Failed to create input tokens.\n");
			free(tokenIds);
			continue;
		}
		free(tokenIds);

		int					   startPos = 0;
		lyGenerationStepResult result;

		printf("Assistant: ");
		fflush(stdout);

		while (true)
		{
			if (!lyGenerateNextToken(&result, pInference, pInputTokens, startPos))
			{
				printf("\nError during token generation.\n");
				break;
			}

			char* decodedText = NULL;
			if (!lyDetokenize(&decodedText, pInference->tokenizer, &result.tokenId, 1))
			{
				printf("\nError decoding token.\n");
				break;
			}

			printf("%s", decodedText);
			fflush(stdout);
			free(decodedText);

			if (result.state != GSInProgress)
			{
				printf("\n");
				break;
			}

			startPos++;
		}

		lyDestroyTensor(pInputTokens);
	}

	lyDestroyInference(pInference);
	lyDestroyModel(pModel);
	printf("\nModel freed successfully!\n");

	return 0;
}