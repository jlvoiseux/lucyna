#include "lyAttention.h"
#include "lyInference.h"
#include "lyModelLoader.h"
#include "lyUtil.h"

#include <stdio.h>

static bool getInput(char* buffer)
{
	printf("%s", "Enter your message (or press enter to quit): ");
	fflush(stdout);
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

	lyUtilPrintDeviceInfo();

	lyModel* pModel;
	lyModelLoaderLoadModel(&pModel, argv[1]);

	printf("Model loaded successfully.\n\n");

	lyInference* pInference;
	lyInferenceCreate(&pInference, pModel, 50, logCallback, argv[1]);

	int32_t* pTokenIds;
	int32_t	 tokenCount;
	lyTokenizerTokenizePrompt(&pTokenIds, &tokenCount, pInference->tokenizer, "Tu es de Gaulle", "Parle de Trump.");

	// Create padded token array
	int32_t* paddedTokens = (int32_t*)malloc(pInference->sequenceLength * sizeof(int32_t));
	for (int32_t i = 0; i < pInference->sequenceLength; i++)
	{
		paddedTokens[i] = pInference->tokenizer->padId;
	}
	memcpy(paddedTokens, pTokenIds, tokenCount * sizeof(int32_t));

	int					   startPos = 0;
	lyGenerationStepResult result;
	int32_t				   currentCount = tokenCount;

	while (true)
	{
		lyInferenceGenerateNextToken(&result, pInference, paddedTokens, &currentCount, startPos);
		startPos = currentCount - 1;

		char* decodedText = NULL;
		lyTokenizerDecodeToken(&decodedText, pInference->tokenizer, result.tokenId);
		printf("%s", decodedText);
		fflush(stdout);
		free(decodedText);

		if (result.state != GSInProgress)
		{
			printf("\n");
			break;
		}
	}

	free(paddedTokens);

	lyInferenceDestroy(pInference);
	lyModelLoaderDestroyModel(pModel);
	printf("\nModel freed successfully!\n");

	free(pTokenIds);

	return 0;
}