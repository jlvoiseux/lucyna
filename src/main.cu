#include "lyAttention.h"
#include "lyModelLoader.h"
#include "lyTokenizerLoader.h"

#include <stdio.h>

static void printDeviceInfo(void)
{
	int			deviceCount;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);
	if (error != cudaSuccess)
	{
		printf("CUDA Error: Failed to get device count: %s\n", cudaGetErrorString(error));
		return;
	}

	printf("CUDA Devices Found: %d\n\n", deviceCount);

	for (int i = 0; i < deviceCount; i++)
	{
		cudaDeviceProp props;
		error = cudaGetDeviceProperties(&props, i);
		if (error != cudaSuccess)
		{
			printf("CUDA Error: Failed to get device properties: %s\n", cudaGetErrorString(error));
			continue;
		}

		printf("Device %d: %s\n", i, props.name);
		printf("  Compute Capability: %d.%d\n", props.major, props.minor);
		printf("  Total Global Memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
		printf("  Memory Clock Rate: %.0f MHz\n", props.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width: %d bits\n", props.memoryBusWidth);
		printf("  L2 Cache Size: %d bytes\n\n", props.l2CacheSize);
	}
}

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

	char prompt[2048];
	while (getInput(prompt))
	{
		if (strlen(prompt) == 0)
		{
			break;
		}

		char formattedPrompt[4096];
		snprintf(formattedPrompt,
				 sizeof(formattedPrompt),
				 "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|>"
				 "<|start_header_id|>assistant<|end_header_id|>\n\n",
				 prompt);

		// TODO: Call inference functions here when implemented
		printf("Model would process: %s\n", formattedPrompt);
	}

	lyDestroyModel(pModel);
	printf("\nModel freed successfully!\n");

	return 0;
}