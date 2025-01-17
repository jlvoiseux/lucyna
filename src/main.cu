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
		printf("  Max Threads per Block: %d\n", props.maxThreadsPerBlock);
		printf("  Max Block Dimensions: %dx%dx%d\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
		printf("  Max Grid Dimensions: %dx%dx%d\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
		printf("  Memory Clock Rate: %.0f MHz\n", props.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width: %d bits\n", props.memoryBusWidth);
		printf("  L2 Cache Size: %d bytes\n\n", props.l2CacheSize);
	}
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Usage: %s <model_dir>\n", argv[0]);
		printf("Example: %s models/Llama3.2-1B-Instruct\n", argv[0]);
		return 1;
	}

	printDeviceInfo();

	lyModel* pModel;
	if (!lyLoadModel(&pModel, argv[1], true, true))
	{
		printf("Failed to load model from directory: %s\n", argv[1]);
		return 1;
	}

	printf("Model configuration:\n");
	printf("  Dimension: %d\n", pModel->args.dim);
	printf("  Layers: %d\n", pModel->args.nLayers);
	printf("  Attention heads: %d\n", pModel->args.nHeads);
	printf("  KV heads: %d\n", pModel->args.nKVHeads);
	printf("  Vocabulary size: %d\n", pModel->args.vocabSize);
	printf("  Multiple of: %d\n", pModel->args.multipleOf);
	printf("  FFN dim multiplier: %.1f\n", pModel->args.ffnDimMultiplier);
	printf("  Norm epsilon: %.1e\n", pModel->args.normEps);
	printf("  Use scaled rope: %s\n", pModel->args.useScaledRope ? "true" : "false");
	printf("  Rope theta: %.1f\n", pModel->args.ropeTheta);
	printf("  Max sequence length: %d\n", pModel->args.maxSequenceLength);
	printf("  N rep: %d\n", pModel->args.nRep);
	printf("  Head dimension: %d\n", pModel->args.headDim);

	printf("\nTensors loaded: %d\n", pModel->tensorCount);
	for (size_t i = 0; i < pModel->tensorCount; i++)
	{
		lyTensor* tensor = &pModel->tensors[i];

		char shapeStr[256] = "[";
		int	 pos		   = 1;
		for (int32_t d = 0; d < tensor->rank; d++)
		{
			if (d == 0)
			{
				pos += snprintf(shapeStr + pos, sizeof(shapeStr) - pos, "%d", tensor->shape[d]);
			}
			else
			{
				pos += snprintf(shapeStr + pos, sizeof(shapeStr) - pos, ",%d", tensor->shape[d]);
			}
		}
		snprintf(shapeStr + pos, sizeof(shapeStr) - pos, "]");

		printf("  %-40s shape: %-20s size: %zu bytes\n", tensor->name, shapeStr, tensor->dataSize);
	}

	lyTokenizer* pTokenizer;
	if (!lyLoadTokenizer(&pTokenizer, argv[1]))
	{
		printf("Failed to load tokenizer from directory: %s\n", argv[1]);
		lyDestroyModel(pModel);
		return 1;
	}
	printf("\nTokenizer loaded with %d tokens\n", pTokenizer->tokenCount);

	lyDestroyModel(pModel);
	printf("\nModel freed successfully!\n");

	return 0;
}