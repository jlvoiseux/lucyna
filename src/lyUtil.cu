#include "lyUtil.h"

#include <stdio.h>
#include <stdlib.h>

unsigned char* lyBase64Decode(const char* input, size_t inLen, size_t* outLen)
{
	static unsigned char	   decode_table[256] = {0};
	static const unsigned char base64_table[65]	 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

	static bool initialized = false;
	if (!initialized)
	{
		memset(decode_table, 0xFF, sizeof(decode_table));
		for (int i = 0; i < 64; i++)
		{
			decode_table[base64_table[i]] = i;
		}
		initialized = true;
	}

	size_t pad = 0;
	if (inLen > 0)
	{
		if (input[inLen - 1] == '=')
			pad++;
		if (inLen > 1 && input[inLen - 2] == '=')
			pad++;
	}

	*outLen				  = (inLen * 3) / 4 - pad;
	unsigned char* output = (unsigned char*)malloc(*outLen + 1);
	if (!output)
		return NULL;

	size_t	 i = 0, j = 0;
	uint32_t accum = 0;
	int		 bits  = 0;

	for (i = 0; i < inLen; i++)
	{
		if (input[i] == '=')
			break;

		unsigned char val = decode_table[(unsigned char)input[i]];
		if (val == 0xFF)
			continue;

		accum = (accum << 6) | val;
		bits += 6;

		if (bits >= 8)
		{
			bits -= 8;
			if (j < *outLen)
			{
				output[j++] = (accum >> bits) & 0xFF;
			}
		}
	}

	output[*outLen] = '\0';
	return output;
}
void lyUtilPrintDeviceInfo(void)
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