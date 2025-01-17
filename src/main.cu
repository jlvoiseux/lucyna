#include "lyModelLoader.h"
#include "lyPickle.h"

#include <stdio.h>

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("Usage: %s <model_dir>\n", argv[0]);
		printf("Example: %s models/Llama3.2-1B-Instruct\n", argv[0]);
		return 1;
	}

	lyModel* pModel;
	if (!lyLoadModel(&pModel, argv[1], true, true))
	{
		printf("Failed to load model from directory: %s\n", argv[1]);
		return 1;
	}

	printf("\nModel loaded successfully!\n");
	printf("Model configuration:\n");
	printf("  Dimension: %d\n", pModel->args.dim);
	printf("  Layers: %d\n", pModel->args.nLayers);
	printf("  Attention heads: %d\n", pModel->args.nHeads);
	printf("  KV heads: %d\n", pModel->args.nKVHeads);
	printf("  Vocabulary size: %d\n", pModel->args.vocabSize);

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

	lyDestroyModel(pModel);
	printf("\nModel freed successfully!\n");

	return 0;
}